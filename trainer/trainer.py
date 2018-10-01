"""
The highest module of model trainer

It gives abstract interface
"""
import torch


class GenerativeModelTrainer:
    """
    Interface of generative model trainer
    """

    def __init__(self,
                 model,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 model_name=None,
                 **kwargs):
        """
        @param model: Model Class (not instance!)
        @param train_loader: Train dataloader
        @param val_loader: Validation dataloader (optional)
        @param test_loader: Test dataloader (optional)
        @param model_name: model_name to save
            (optional, default for model class name)
        @param **kwargs: keyword arguments for model constructor
        """
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.model = model(**kwargs).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        if model_name is None:
            self.model_name = self.model.__class__.__name__
        else:
            self.model_name = model_name

    def update_optimizer(self):
        """
        Update model parameter via optimizer
        """
        raise NotImplementedError

    def calculate_loss(self):
        """
        Calculate GAN loss of model
        """
        raise NotImplementedError

    def load_pretrained_model(self, *args):
        """
        Load pretrained model

        @param *args: iterable of filenames of pretrained state_dict (string)
        """
        raise NotImplementedError

    def save_model_state(self):
        """
        Save current model state
        """
        raise NotImplementedError
