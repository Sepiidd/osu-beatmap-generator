import h5py
from torch.utils.data import Dataset

class OBGMapDataset(Dataset):
    def __init__(self, h5path):
        self.h5path = h5path
        self.num_samples = self.get_len(h5path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5path, 'r') as f:
            #TODO
            pass

    def get_len(path):
        with h5py.File(path, 'r') as f:
            return f.attrs.get("num_samples")
