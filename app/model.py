import torch
import segmentation_models_pytorch as smp 
from app.config import N_CHANNELS, NUM_CLASSES, CHECKPOINT_PATH

def load_model():
    model = smp.Unet(
        encoder_name = 'resnet34',
        encoder_weights = None, 
        in_channels = N_CHANNELS,
        classes = NUM_CLASSES,
        activation = None
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model