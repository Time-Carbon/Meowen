import os

os.environ["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import unsloth
import torch

torch.cuda.set_per_process_memory_fraction(1.0, device=0)

from .trainer import *
from .datasetLoader import *
from .modelLoader import *
