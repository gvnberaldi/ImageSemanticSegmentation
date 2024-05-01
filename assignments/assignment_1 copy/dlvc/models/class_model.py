import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''
        if suffix is not None:
            save_dir = Path.joinpath(save_dir, suffix)

        torch.save(self.state_dict(), save_dir)

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        
        self.load_state_dict(torch.load(path,
                                        map_location=torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')))