import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from librosa import frames_to_time

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
        #TODO: debug, fix target generation
        sample, _ = self.sample_bin_search(idx)
        audio_feat = sample["audio_feat"]
        audio_targets = sample["audio_targets"]
        num_frames = audio_feat.shape[1]

        if num_frames < self.max_seq_len: #edge, audio less than max_seq_len
            audio_feat = self.pad_feats(audio_feat)
            num_frames = audio_feat.shape[1]
        print("frame count is", num_frames)

        start_idx = np.random.randint(0, num_frames-self.max_seq_len+1) 
        print("selected start_idx is", start_idx)
        start_time = frames_to_time(start_idx, sr=SR, hop_length=HOP_LEN)*1000
        print("time (ms) of start_idx is", start_time)

        window_seq = self.slice_windows(audio_feat, start_idx)
        targets = self.gen_targets(start_time, audio_feat, audio_targets)

        return window_seq, targets

    def gen_targets(self, start_time, audio_feat, audio_targets):
        '''
        generates sequence of 0,1 representing negative and positive targets for each 10=(HOP_LEN/SR)*1000 milliseconds  found in <audio_feat>, starting from <start_time>
        '''
        #TODO
        pass

    def slice_windows(self, audio_feat, start_idx):
        '''
        retrieve <self.max_seq_len> many 15x80x3 (centered input frame+context, mel frequency, window length) slices of <audio_feat>, starting from <start_idx>
        '''
        slices = []
        i = 0
        while i < self.max_seq_len:
            slices.append(self.retrieve_window(audio_feat, start_idx+i))
        stacked = np.stack(slices, axis=0)
        return stacked 

    def retrieve_window(self, audio_feat, idx):
        '''
        returns a 15x80x3 slice of <audio_feat> at index <idx>, pads with 0s along the size 80 frequency dimension if too close to start/end
        '''
        #TODO
        pass

    def pad_feats(self, audio_feat):
        '''
        pads the end of <audio_feat> to size <self.max_seq_len> with 0's along the size 80 frequency dimension
        '''
        #TODO
        pass

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
    h5path = BASE_DIR.parent.parent / "datasets" / "partition0"
    dataset = OBGAudioDataset(h5path, SEQUENCE_LEN)
    dataset.__getitem__(23)
    print("done")
