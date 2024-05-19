import torch
import torch.nn as nn
from pathlib import Path

class DeepSegmenter(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''

        if suffix is None:
            path = save_dir / f'{self.net._get_name()}_model.pth'
        else:
            path = save_dir / f'{self.net._get_name()}_model_{suffix}.pth'

        torch.save(self.net.state_dict(), path)

    def load(self, path: Path, suffix=None):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        if suffix is None:
            path = path / f'{self.net._get_name()}_model.pth'
        else:
            path = path / f'{self.net._get_name()}_model_{suffix}.pth'


        state_dict = torch.load(path, map_location='cpu')

        local_dict = self.net.state_dict()

        for key, value in state_dict.items():
            if key not in local_dict:
                 continue
            if 'encoder' in key:  # only load encoder weights
                local_dict[key].copy_(value)
        self.net.load_state_dict(local_dict)

    def load_all_weights(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device('gpu') if torch.cuda.is_available()
                                                                               else torch.device('cpu')))
        