import torch
import torch.nn as nn

class QuasiLikelihoodLoss(nn.Module):
    """
    Implementation of the Quasi-Likelihood Loss Function
    for volatility forecasting.
    """
    def __init__(self):
        super(QuasiLikelihoodLoss, self).__init__()
    
    def forward(self, predicted_var, true_var):
        """
        Compute the Quasi-Likelihood Loss.
        
        Parameters:
        ----------
        predicted_var : torch.Tensor
            The predicted variance (or squared volatility) tensor. Shape: (batch_size, ...)
        true_var : torch.Tensor
            The true variance (or squared volatility) tensor. Shape: (batch_size, ...)
        
        Returns:
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Ensure no division by zero (add epsilon for numerical stability)

        predicted_var = predicted_var 
        true_var = true_var 
        
        # Compute the loss
        ratio = predicted_var / true_var
        loss = ratio - torch.log(ratio) - 1
        
        # Return the mean loss across all samples
        return loss.mean()