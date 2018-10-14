import unittest
from utils import AverageTracker


class UtilTest(unittest.TestCase):
    def test_average_tracker(self):
        tracker = AverageTracker('test')
        for i in range(10):
            tracker.update(i, 1)
        self.assertLessEqual(tracker() - 4.5, 1e-4)
