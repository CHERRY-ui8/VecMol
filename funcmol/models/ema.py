# inspired by https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
from copy import deepcopy

import torch
import torch.nn as nn


class ModelEma(nn.Module):
    """
    ModelEma is a class that maintains an exponential moving average (EMA) of a given model's parameters.

    Attributes:
        module (nn.Module): A copy of the input model used to accumulate the moving average of weights.
        decay (float): The decay rate for the EMA. Default is 0.9999.

    Methods:
        update(model):
            Updates the EMA parameters using the current parameters of the given model.

        set(model):
            Sets the EMA parameters to be exactly the same as the given model's parameters.
    """
    def __init__(self, model, decay=0.9999):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        # Get device from model parameters (PyTorch models don't have a .device attribute)
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        self.to(device)

    def _update(self, model, update_fn):
        """
        Update EMA parameters.
        
        Args:
            model: The model to update from. Can be wrapped in DDP.
            update_fn: Function to compute new EMA value: (ema_v, model_v) -> new_value
        """
        # Unwrap model if it's wrapped in DDP
        if hasattr(model, 'module'):
            model = model.module
        
        with torch.no_grad():
            ema_state_dict = self.module.state_dict()
            model_state_dict = model.state_dict()
            
            # Iterate through keys to ensure matching
            for key in ema_state_dict.keys():
                if key not in model_state_dict:
                    continue
                
                ema_v = ema_state_dict[key]
                model_v = model_state_dict[key]
                
                # Skip None values
                if ema_v is None or model_v is None:
                    continue
                
                # Skip non-tensor values
                if not isinstance(ema_v, torch.Tensor) or not isinstance(model_v, torch.Tensor):
                    continue
                
                # Ensure shapes match
                if ema_v.shape != model_v.shape:
                    continue
                
                # Update EMA value
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the EMA model.
        """
        return self.module(*args, **kwargs)