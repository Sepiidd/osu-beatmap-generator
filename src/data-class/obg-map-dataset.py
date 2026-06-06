import h5py
from torch.utils.data import Dataset

class OBGMapDataset(Dataset):
    def __init__(self, h5path):
        self.h5path = h5path
        self.num_samples = self.get_len(h5path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sample_bin_search(idx)

    def sample_bin_search(self, target):
        with h5py.File(self.h5path, 'r') as f:
            song_lst = list(f.keys())
            p1 = 0
            p2 = len(song_lst)-1
            curr_name = None
            sample = {}
            found = False
            while p1 <= p2 and not found:
                curr = (p1+p2)//2
                curr_name = song_lst[curr]
                grp_set = f.get(curr_name)
                start_range = grp_set.attrs.get("start_range")
                end_range = grp_set.attrs.get("end_range")
                if start_range <= target <= end_range:
                    grp = grp_set.get(str(target))
                    audio_feat = grp_set.get("audio_feat")[:]
                    audio_targets = grp.get("audio_targets")[:]
                    osu_tokens = grp.get("osu_tokens")[:]
                    deltas_fwd = grp.get("deltas_fwd")[:]
                    deltas_back = grp.get("deltas_back")[:]
                    sample = {
                            "audio_feat": audio_feat,
                            "audio_targets": audio_targets,
                            "osu_tokens": osu_tokens,
                            "deltas_fwd": deltas_fwd,
                            "deltas_back": deltas_back
                            }
                    found = True
                elif target < start_range:
                    p2 = curr-1
                elif target > end_range:
                    p1 = curr+1
            return (sample, found)

    def get_len(self, path):
        with h5py.File(path, 'r') as f:
            return f.attrs.get("num_samples")
