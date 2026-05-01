import numpy as np
import os

N_CHANNELS = 4
NUM_CLASSES = 3
PATCH_SIZE = 256

# Global Normalization Stats (Calculated from Colab)
MEAN = np.array([0.05420399, 0.07465479, 0.06314979, 0.29062513], dtype=np.float32)
STD = np.array([0.02797917, 0.02933381, 0.03717266, 0.12625116], dtype=np.float32)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Make sure you place the new model file in the checkpoints folder!
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model_v2.pth")
