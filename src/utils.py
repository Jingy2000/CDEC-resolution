import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LossTracker:
    def __init__(self):
        self.history = defaultdict(list)
        
    def update(self, metrics):
        for k, v in metrics.items():
            self.history[k].append(v)
            
    def plot(self):
        plt.figure(figsize=(10, 6))
        for metric, values in self.history.items():
            plt.plot(values, label=metric)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
