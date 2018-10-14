"""
Average Tracker for training
"""


class AverageTracker:
    """
    Average Tracker implementation for loss
    """

    def __init__(self, name):
        self._name = name
        self.value = 0
        self.n = 0
        pass

    def __call__(self):
        return self.get_value()

    def __len__(self):
        return self.n

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise AttributeError('Cannot modify name of average tracker')

    def update(self, value, n=1):
        self.value = ((self.value * self.n) + (value * n)) / (self.n + n)
        self.n += n

    def get_value(self):
        return self.value

    def initialize(self):
        self.value = 0
        self.n = 0
