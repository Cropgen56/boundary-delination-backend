import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from app.config import CHECKPOINT_PATH


class TripleHeadModel(nn.Module):
    """
    UNet with EfficientNet-B4 encoder and three separate output heads:
      - extent_head   : binary field/no-field probability (sigmoid)
      - boundary_head : binary field boundary probability (sigmoid)
      - distance_head : per-pixel distance to nearest boundary (raw logits)

    Inputs : (B, 3, H, W) float32 — normalised RGB imagery
    Outputs: (extent, boundary, distance) each (B, 1, H, W)
    """

    def __init__(self,
                 encoder_name: str = 'efficientnet-b4',
                 encoder_weights: str | None = 'imagenet',
                 dropout_p: float = 0.2):
        super().__init__()
        base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        self.encoder  = base.encoder
        self.decoder  = base.decoder

        dc = base.decoder.blocks[-1].conv2[0].out_channels   # decoder output channels

        self.extent_dropout   = nn.Dropout2d(dropout_p)
        self.boundary_dropout = nn.Dropout2d(dropout_p)

        self.extent_head   = nn.Conv2d(dc, 1, kernel_size=3, padding=1)
        self.boundary_head = nn.Conv2d(dc, 1, kernel_size=3, padding=1)
        self.distance_head = nn.Conv2d(dc, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        features    = self.encoder(x)
        decoder_out = self.decoder(*features)

        extent   = self.extent_head(self.extent_dropout(decoder_out))
        boundary = self.boundary_head(self.boundary_dropout(decoder_out))
        distance = self.distance_head(decoder_out)

        return extent, boundary, distance


def load_model() -> TripleHeadModel:
    """Load fine-tuned TripleHeadModel from checkpoint. encoder_weights=None so
    imagenet weights are NOT downloaded at startup — only the saved state dict
    is used."""
    model = TripleHeadModel(encoder_weights=None)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model