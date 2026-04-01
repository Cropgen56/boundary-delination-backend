import torch
import segmentation_models_pytorch as smp 
from config import N_CHANNELS, CHECKPOINT_PATH

def load_model():
    model = smp.Unet(
        encoder_name = 'resnet34',
        encoder_weights = None, 
        in_channels = N_CHANNELS,
        classes = 1,
        activation = None
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model 