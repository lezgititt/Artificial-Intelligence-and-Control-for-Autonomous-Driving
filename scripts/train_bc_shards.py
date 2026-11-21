"""Memory-bounded BC training on episode NPZ shards, with a wall-time limit."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_bc_actor import lane_class, export_policy, epoch, initialize_from_raw_policy
from utils.bc_results import create_bc_result_dir
from utils.networks import HybridActorCritic
from train import HPARAMS


def load(path):
    with np.load(path,allow_pickle=False) as d:
        return d['observation'],d['action_raw']


def run_pass(model, paths, mean, std, weights, batch_size, optimizer=None, deadline=None):
    total_n=0; squared=0.;ce=0.;denom=0.;conf=np.zeros((3,3),np.int64)
    completed=True
    for path in paths:
        if deadline is not None and time.monotonic()>=deadline:
            completed=False;break
        obs,act=load(path);labels=lane_class(act)
        x=(torch.as_tensor(obs,device=mean.device)-mean)/std
        y=torch.as_tensor(act[:,0],device=mean.device)
        z=torch.as_tensor(labels,device=mean.device)
        result=epoch(model,x,y,z,weights,batch_size,optimizer)
        n=len(obs);d=float(weights[z].sum().item())
        total_n+=n;squared+=result['accel_mse']*n;ce+=result['lane_ce']*d;denom+=d
        conf+=np.asarray(result['confusion'])
    if not total_n:
        return None
    diag=conf.diagonal();precision=diag/np.maximum(conf.sum(0),1);recall=diag/np.maximum(conf.sum(1),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return dict(loss=squared/total_n+ce/denom,accel_mse=squared/total_n,lane_ce=ce/denom,
                lane_macro_f1=float(f1.mean()),left_precision=float(precision[0]),left_recall=float(recall[0]),
                right_precision=float(precision[2]),right_recall=float(recall[2]),confusion=conf.tolist(),
                samples=total_n,complete_pass=completed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_dir',type=Path);p.add_argument('--out-dir', type=Path, help='Default: results/BC_YYYYMMDD_HHMMSS')
    p.add_argument('--wall-seconds',type=float,default=21600);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=8192);p.add_argument('--lr',type=float,default=4.2e-4)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda');p.add_argument('--patience',type=int,default=100)
    p.add_argument('--hidden-sizes', nargs='+', type=int, default=HPARAMS['hidden_sizes'])
    p.add_argument('--init-model', type=Path)
    p.add_argument('--lane-weight-root', type=int, default=2,
                   help='Root of inverse lane class frequency (default: 2)')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.lane_weight_root < 1:
        p.error('--lane-weight-root must be at least 1')
    manifest=json.loads((a.data_dir/'manifest.json').read_text());collection=json.loads((a.data_dir/'config.json').read_text())
    paths=[a.data_dir/x for x in manifest['shards']];rng=np.random.default_rng(a.seed)
    permutation=rng.permutation(len(paths));nval=max(1,round(.05*len(paths)))
    val_indices=set(manifest.get("validation_indices", map(int, permutation[:nval])));train=[x for i,x in enumerate(paths) if i not in val_indices];val=[x for i,x in enumerate(paths) if i in val_indices]
    if len(train)==0:raise ValueError('Need independent train/validation episodes')
    # Merge per-shard moments without retaining all observations in RAM/GPU.
    count=0;mean=np.zeros(manifest['obs_dim'],np.float64);m2=np.zeros_like(mean);counts=np.zeros(3,np.int64)
    for path in train:
        obs,act=load(path);n=len(obs);local=obs.mean(0,dtype=np.float64);delta=local-mean
        m2+=np.square(obs-local).sum(0)+delta**2*count*n/(count+n)
        mean+=delta*n/(count+n);count+=n;counts+=np.bincount(lane_class(act),minlength=3)
    if (counts==0).any():raise ValueError(f'Missing lane class: {counts}')
    std=np.sqrt(np.maximum(m2/count,0));std=np.where(std<1e-6,1,std)
    mean=mean.astype(np.float32);std=std.astype(np.float32)
    w = np.power(count / (3 * counts), 1 / a.lane_weight_root)
    w /= w.mean()
    a.out_dir = create_bc_result_dir(a.out_dir)
    device=torch.device(a.device);mt=torch.tensor(mean,device=device);st=torch.tensor(std,device=device);wt=torch.tensor(w,dtype=torch.float32,device=device)
    model=HybridActorCritic(manifest['obs_dim'],hidden_sizes=a.hidden_sizes,activation=HPARAMS['activation'],init_log_std=HPARAMS['init_log_std'],lane_keep_bias=HPARAMS['lane_keep_bias']).to(device)
    if a.init_model:
        state = torch.load(a.init_model, map_location=device, weights_only=True)
        initialize_from_raw_policy(model, state, mt, st)
    else:
        with torch.no_grad():model.v_head.weight.zero_();model.v_head.bias.zero_()
    for param in model.v_head.parameters():param.requires_grad_(False)
    model.log_std.requires_grad_(False)
    optimizer=torch.optim.Adam([x for x in model.parameters() if x.requires_grad],lr=a.lr)
    config=dict(data=str(a.data_dir/'manifest.json'),dataset_directory=str(a.data_dir.resolve()),init_model=str(a.init_model) if a.init_model else None,
                algorithm='supervised_bc_on_ppo_hybrid_actor',observation_dim=manifest['obs_dim'],hidden_sizes=a.hidden_sizes,activation=HPARAMS['activation'],seed=a.seed,
                observation_neighbor_distance_mode=collection['observation'].get('neighbor_distance_mode','legacy'),
                collision_detection=collection.get('collision_detection'),
                epochs=a.epochs,wall_seconds=a.wall_seconds,batch_size=a.batch_size,lr=a.lr,train_samples=count,
                validation_samples=manifest['samples']-count,train_episodes=[i for i in range(len(paths)) if i not in val_indices],validation_episodes=sorted(val_indices),
                lane_class_order=['left','keep','right'],train_lane_counts=counts.tolist(),lane_weights=w.tolist(),
                lane_weight_root=a.lane_weight_root,
                loss='accel_mse + inverse_frequency_lane_ce',storage='episode_npz_shards',
                collection_manifest_sha256=hashlib.sha256((a.data_dir/'manifest.json').read_bytes()).hexdigest())
    (a.out_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n');np.savez(a.out_dir/'normalization.npz',mean=mean,std=std)
    for source in ['env/road_config.py','env/mdp_config.py','train.py','scripts/train_bc_shards.py','env/ring_observation.py']:
        shutil.copy2(source,a.out_dir/Path(source).name)
    with (a.out_dir/'mdp_config.py').open('a') as snapshot:
        snapshot.write('\nOBSERVATION = '+repr(collection['observation'])+'\nSIMULATION = '+repr(collection['simulation'])+'\n')
    history=[];best=float('inf');best_epoch=0;start=time.monotonic();deadline=start+a.wall_seconds
    fields=['epoch','train_loss','val_loss','val_accel_mse','val_lane_ce','val_lane_macro_f1','val_left_precision','val_left_recall','val_right_precision','val_right_recall']
    with SummaryWriter(str(a.out_dir/'tb')) as writer,(a.out_dir/'training_log.csv').open('w') as f:
        csvwriter=csv.DictWriter(f,fieldnames=fields);csvwriter.writeheader()
        for e in range(1,a.epochs+1):
            shuffled=[train[i] for i in rng.permutation(len(train))]
            tr=run_pass(model,shuffled,mt,st,wt,a.batch_size,optimizer,deadline)
            if tr is None:break
            va=run_pass(model,val,mt,st,wt,a.batch_size)
            row=dict(epoch=e,train_loss=tr['loss'],val_loss=va['loss'],**{'val_'+k:va[k] for k in ['accel_mse','lane_ce','lane_macro_f1','left_precision','left_recall','right_precision','right_recall']})
            csvwriter.writerow(row);f.flush()
            for k,v in row.items():
                if k!='epoch':writer.add_scalar(k,v,e)
            history.append(dict(epoch=e,train=tr,validation=va))
            exported=export_policy(model,mt,st)
            torch.save(exported,a.out_dir/'last_model.pt')
            if va['loss']<best:
                best=va['loss'];best_epoch=e;torch.save(exported,a.out_dir/'model.pt')
                torch.save(dict(state_dict=model.state_dict(),obs_mean=mt,obs_std=st,epoch=e),a.out_dir/'bc_training_state.pt')
            result=dict(status='complete' if e==a.epochs or time.monotonic()>=deadline or e-best_epoch>=a.patience else 'training',best_epoch=best_epoch,best_val_loss=best,epochs_completed=e,wall_seconds=time.monotonic()-start,best_validation=history[best_epoch-1]['validation'],init_model=str(a.init_model) if a.init_model else None)
            (a.out_dir/'metrics.json').write_text(json.dumps(history,indent=2)+'\n');(a.out_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'Epoch {e}: train={tr["loss"]:.6f} val={va["loss"]:.6f} complete={tr["complete_pass"]} seconds={time.monotonic()-start:.1f}',flush=True)
            if time.monotonic()>=deadline or e-best_epoch>=a.patience:break
    # Export parity is verified on an independent validation shard.
    saved=torch.load(a.out_dir/'bc_training_state.pt',map_location=device,weights_only=True);model.load_state_dict(saved['state_dict']);model.eval()
    raw=torch.tensor(load(val[0])[0][:1024],device=device);raw_model=deepcopy(model);raw_model.load_state_dict(torch.load(a.out_dir/'model.pt',map_location=device,weights_only=True));raw_model.eval()
    with torch.no_grad():torch.testing.assert_close(model.det_action((raw-mt)/st),raw_model.det_action(raw),rtol=1e-4,atol=1e-5)
    result['status']='complete';result['raw_input_export_verified']=True
    (a.out_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
