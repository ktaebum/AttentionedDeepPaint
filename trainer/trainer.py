"""
Abstract for model trainer
"""


class ModelTrainer:
    def __init__(self, args, data_loader, device):
        """
        Constructor for ModelTrainer

        @param args: parsed argument
        @param data_loader: Data Loader for model
        @param device: torch device (cuda or cpu)
        """

        self.args = args
        self.data_loader = data_loader
        self.device = device
        self.resolution = 512

    def train(self):
        raise NotImplementedError

    def validate(self, dataset, epoch, samples=3):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save_model(self, name, epoch):
        raise NotImplementedError

    def _set_optimizers(self):
        raise NotImplementedError

    def _set_losses(self):
        raise NotImplementedError

    def _update_generator(self):
        raise NotImplementedError

    def _update_discriminator(self):
        raise NotImplementedError
