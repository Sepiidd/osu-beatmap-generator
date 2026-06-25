import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from librosa import frames_to_time
from preprocess.audio_utils import get_frames_at_idx

#GLOBALS
BASE_DIR = Path(__file__).parent
SR = 22050
SEQUENCE_LEN = 512 #equal to roughly 5 seconds of audio
HOP_LEN=221

class OBGAudioDataset(Dataset):
    def __init__(self, h5path, max_seq_len):
        self.h5path = h5path
        self.num_samples = self.get_len(h5path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample, _ = self.sample_bin_search(idx)
        audio_feat = sample["audio_feat"]
        audio_targets = sample["audio_targets"]
        num_frames = audio_feat.shape[1]

        if num_frames < self.max_seq_len: #edgecase, audio less than max_seq_len
            audio_feat = self.pad_feats(audio_feat)
            num_frames = audio_feat.shape[1]

        start_idx = np.random.randint(0, num_frames-self.max_seq_len+1) 

        window_seq = self.slice_windows(audio_feat, start_idx)
        window_seq = np.swapaxes(window_seq, 1, 2) #swap axes bc im dumb
        targets = self.gen_targets(start_idx, audio_targets)

        return torch.tensor(window_seq), torch.tensor(targets)

    def gen_targets(self, start_idx, audio_targets, lenience=20):
        '''
        generates sequence of 0,1 representing negative and positive targets for each 10=(HOP_LEN/SR)*1000 milliseconds  found in <audio_feat>, starting from <start_time>
        '''
        ms_per_frame = (HOP_LEN/SR)*1000
        targets = []
        time_ms = frames_to_time(start_idx, sr=SR, hop_length=HOP_LEN)*1000
        targets_idx = self.bin_search_closest(time_ms, audio_targets, lenience) 
        curr_target = audio_targets[targets_idx]
        for i in range(self.max_seq_len):
            time_ms = frames_to_time(start_idx+i, sr=SR, hop_length=HOP_LEN)*1000
            t = 0
            if abs(curr_target-time_ms) < lenience:
                targets_idx = targets_idx+1 if targets_idx+1 <= len(audio_targets)-1 else 0 #set to 0 if last target passed
                curr_target = audio_targets[targets_idx]
                t = 1
            targets.append(t)
        return np.array(targets)

    def slice_windows(self, audio_feat, start_idx):
        '''
        retrieve <self.max_seq_len> many 15x80x3 (centered input frame+context, mel frequency, window length) slices of <audio_feat>, starting from <start_idx>
        '''
        slices = []
        i = 0
        while i < self.max_seq_len:
            slices.append(get_frames_at_idx(audio_feat, start_idx+i))
            i += 1
        stacked = np.stack(slices, axis=0)
        return stacked 

    def pad_feats(self, audio_feat):
        '''
        pads the end of <audio_feat> to size <self.max_seq_len> with 0's along the size 80 frequency dimension
        '''
        curr_len = audio_feat.shape[1]
        to_pad = self.max_seq_len - curr_len
        padded = np.pad(audio_feat, ((0, 0), (0, to_pad), (0, 0)))
        return padded

    def bin_search_closest(self, target, lst, lenience=20):
        '''
        return index of closest element (prefer lower index) to <target> within <lst>
        *literally just binary search but more lenient on pointer assignment
        '''
        if len(lst) == 0:
            return None
        if len(lst) <= 2:
            return 0

        p1 = 0
        p2 = len(lst)-1
        while p1 != p2-1:
            curr = (p1 + p2) // 2
            val = lst[curr]
            if val == target:
                return curr
            elif val < target:
                p1 = curr
            elif val > target:
                p2 = curr
        if target-lst[p1] > lenience:
            return p2
        return p1

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

if __name__ == "__main__":
    #test stuff below if needed
    h5path = BASE_DIR.parent.parent / "datasets" / "train"
    dataset = OBGAudioDataset(h5path, SEQUENCE_LEN)
    window, targets = dataset.__getitem__(23)
    print("window shape:", window.shape)
    print("window type:", window.dtype)
    print("targets shape:", targets.shape)
    print("targets type:", targets.dtype)
