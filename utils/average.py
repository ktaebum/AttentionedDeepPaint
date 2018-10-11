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

    def update(self, value):
        if self.n == 0:
            self.value = value
        else:
            self.value = ((self.value * self.n) + value) / (self.n + 1)
        self.n += 1

    def get_value(self):
        return self.value

    def initialize(self):
        self.value = 0
        self.n = 0
