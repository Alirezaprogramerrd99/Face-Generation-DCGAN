import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.datasets.utils import *

from pathlib import Path
from main_model.data_prepare import *

from main_model.DCGAN_model import *

# Set random seed for reproducibility
# # Set random seeds
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

manualSeed = 48
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

data_path = Path("celeba_dataset")
image_path = data_path / "celeba"

prepare_data(data_path, image_path, "img_align_celeba")
walk_through_dir(image_path / "img_align_celeba")