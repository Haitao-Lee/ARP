# coding = utf-8
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random


seed = 2023
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)