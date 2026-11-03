"""The residual recurrent network, its penalised loss and its training loop."""
from .residual import ResidualRNN, penalised_loss
from .train import TrainConfig, bag_residuals, fit_bag, predict_bag, train_net

__all__ = ["ResidualRNN", "penalised_loss", "TrainConfig", "train_net", "fit_bag", "bag_residuals",
           "predict_bag"]
