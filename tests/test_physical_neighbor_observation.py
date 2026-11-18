"""Keep near vehicles visible and use one physical distance convention."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import xml.etree.ElementTree as ET

from env.sumo_env import SumoHighwayEnv
from env.collision_detection import configure_collision_gaps


class PhysicalNeighborTests(unittest.TestCase):
    def setUp(self):
        self.env = SumoHighwayEnv.__new__(SumoHighwayEnv)
        self.env.ego_id = 'ego'
        self.env.visibility = 50.
        self.vehicle = Mock()
        self.vehicle.getMinGap.side_effect = lambda vid: {'ego': 2.5, 'front': 2., 'rear': 2.}[vid]
        self.vehicle.getSpeed.side_effect = lambda vid: {'front': 0., 'rear': 1.}[vid]

    def test_close_leader_remains_visible_and_old_encoding_is_explicit(self):
        with patch('env.sumo_env.traci', SimpleNamespace(vehicle=self.vehicle)):
            for physical_gap in [2.51, 2.49, 2.0, .21, .01, -.1]:
                self.vehicle.getLeader.return_value = ('front', physical_gap - 2.5)
                gap, speed = self.env._neighbor(0, True, physical=True)
                self.assertAlmostEqual(gap, max(physical_gap, 0))
                self.assertEqual(speed, 0)
            self.assertIsNone(self.env._neighbor(0, True))
            self.vehicle.getLeader.return_value = ('', -1)
            self.assertIsNone(self.env._neighbor(0, True, physical=True))

    def test_rear_gap_uses_followers_min_gap(self):
        self.vehicle.getFollower.return_value = ('rear', -1.7)
        with patch('env.sumo_env.traci', SimpleNamespace(vehicle=self.vehicle)):
            gap, speed = self.env._neighbor(0, False, physical=True)
        self.assertAlmostEqual(gap, .3)
        self.assertEqual(speed, 1)

    def test_adjacent_gap_and_visibility_use_physical_distance(self):
        with patch('env.sumo_env.traci', SimpleNamespace(vehicle=self.vehicle)):
            self.vehicle.getNeighbors.return_value = [('front', -2.3)]
            self.assertAlmostEqual(self.env._neighbor(1, True, physical=True)[0], .2)
            self.vehicle.getNeighbors.return_value = [('rear', -1.6)]
            self.assertAlmostEqual(self.env._neighbor(-1, False, physical=True)[0], .4)
            self.vehicle.getLeader.return_value = ('front', 49.)
            self.assertIsNone(self.env._neighbor(0, True, physical=True))

    def test_two_lane_scan_keeps_overlap_and_does_not_add_min_gap(self):
        self.env._scan_lane_neighbors = lambda offset: ((.3, 0.), (-.1, 1.))
        self.assertEqual(self.env._neighbor(2, True, physical=True), (.3, 0.))
        self.assertEqual(self.env._neighbor(-2, False, physical=True), (0., 1.))


class CollisionGapConfigurationTests(unittest.TestCase):
    def test_all_expert_types_share_gap_without_changing_following_parameters(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'road.sumocfg'
            config.write_text('<configuration><input><route-files value="types.xml"/></input></configuration>')
            path = root / 'types.xml'
            path.write_text('<routes><vType id="ego_type"/><vTypeDistribution id="cars">'
                            '<vType id="car_krauss" minGap="2.5"/>'
                            '<vType id="car_idm" minGap="2.0"/></vTypeDistribution></routes>')
            result = configure_collision_gaps(config, ego_gap_m=.2, background_gap_m=.2)
            self.assertEqual(set(result), {'ego_type', 'car_krauss', 'car_idm'})
            for node in ET.parse(path).getroot().iter('vType'):
                min_gap = float(node.get('minGap', '2.5'))
                self.assertAlmostEqual(min_gap * float(node.get('collisionMinGapFactor')), .2)
            self.assertEqual(result['car_idm']['min_gap_m'], 2.)


if __name__ == '__main__':
    unittest.main()
