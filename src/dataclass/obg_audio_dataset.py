import h5py
import numpy as np
import torch
import traceback
from torch.utils.data import Dataset
from pathlib import Path
from librosa import frames_to_time
from librosa import time_to_frames 
from preprocess.audio_utils import get_frames_at_idx
from configs.audio_config import AudioConfig

#GLOBALS
BASE_DIR = Path(__file__).parent
configA = AudioConfig()
SR = configA.sr
SEQUENCE_LEN = configA.sequence_len #equal to roughly 5 seconds of audio
HOP_LEN = configA.hop_len

class OBGAudioDataset(Dataset):
    def __init__(self, h5path, max_seq_len, augment=False, augmentations=[], benchmark=False, test=False):
        self.h5path = h5path
        self.num_samples = self.get_len(h5path)
        self.max_seq_len = max_seq_len
        self.augment = augment #augmentation bool flag
        self.augmentations = augmentations.copy() #list of functions, for which each takes in inputs and targets, returns augmented inputs and targets
        self.augmentations.append(lambda x, y: (x, y))
        self.benchmark = benchmark
        self.test = test #fix start_idx for testing within getitem()

        self.song_num_frames = None #update per song

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
#        idx = 2060 #NOTE: test
        sample, _ = self.sample_bin_search(idx)
        audio_feat = sample["audio_feat"]
        audio_targets = sample["audio_targets"]
        stars = sample["stars"]
        aim = sample["aim"]
        speed = sample["speed"]

        name = sample["song"]
        d = sample["diff"]
#        print(f"song, diff are {name} and {d}, audio feat has shape {audio_feat.shape}, audio targets shape {audio_targets.shape}")

        if self.augment:
            option_select = np.random.randint(0, len(self.augmentations))
            audio_feat, audio_targets = self.augmentations[option_select](audio_feat, audio_targets)

        num_frames = audio_feat.shape[1]
        self.song_num_frames = num_frames

        if num_frames < self.max_seq_len: #edgecase, audio less than max_seq_len
            audio_feat = self.pad_feats(audio_feat)
            num_frames = audio_feat.shape[1]

        final_ms = audio_targets[-1]
        final_idx = time_to_frames(final_ms/1000, sr=SR, hop_length=HOP_LEN)
        start_limit = final_idx - self.max_seq_len + 1 
        start_limit = 0 if start_limit < 0 else start_limit

        print(f"")
        print(f"final_ms {final_ms}, final_idx {final_idx}, start_limit {start_limit}")
        start_idx = np.random.randint(0, start_limit+1) if not (self.benchmark or self.test) else (num_frames // 2 if self.test else 0)
#        start_idx = np.random.randint(0, num_frames-self.max_seq_len+1) if not self.benchmark else 0 #NOTE: test random start index for onset target comparison
#        start_idx = 19708 #NOTE: test
        print("start idx", start_idx)
        max_len = self.max_seq_len if not self.benchmark else num_frames

        #TODO: see error in /src, debug!!
        try:
            window_seq = self.slice_windows(audio_feat, start_idx, max_len)
            window_seq = np.swapaxes(window_seq, 1, 2) #swap axes bc im dumb
            targets = self.gen_targets(start_idx, audio_targets)
            if self.test:
                return torch.tensor(window_seq), torch.tensor([stars, aim, speed], dtype=torch.float32), torch.tensor(targets), start_idx

            if self.benchmark:
                return torch.tensor(audio_feat), torch.tensor([stars, aim, speed], dtype=torch.float32), torch.tensor(targets)

            return torch.tensor(window_seq), torch.tensor([stars, aim, speed], dtype=torch.float32), torch.tensor(targets)
        except Exception as e:
            print("sample idx, start idx are", idx, start_idx)
            traceback.print_exception(e)

    def gen_targets(self, start_idx, audio_targets):
        '''
        generates sequence of 0,1 representing negative and positive targets for each 10=(HOP_LEN/SR)*1000 milliseconds  found in <audio_feat>, starting from <start_time>
        '''
        #TODO: if selected index is beyond first hitsound, errors get thrown, FIX

        frame_size = (1000/configA.sr) * configA.hop_len
        frame_side = frame_size / 2 #time to edge of frame from center

        max_seq_len = self.max_seq_len if not self.benchmark else self.song_num_frames
        targets = np.zeros(max_seq_len)
        start_time = frames_to_time(start_idx, sr=SR, hop_length=HOP_LEN)*1000 #in milliseconds
        start_time = start_time-frame_side #start time lines up with start of frame
        end_time = frames_to_time(start_idx+max_seq_len, sr=SR, hop_length=HOP_LEN)*1000 #in milliseconds
        end_time = end_time+frame_side #end time lines up with end of frame

        target_idx = self.find_first(start_time, audio_targets)
        curr_target = audio_targets[target_idx] / 1000
        print(f"audio targets size {len(audio_targets)}, find first idx {target_idx} with target {curr_target*1000}")

#        prev = None #NOTE: test
#        prev_idx = None
        while curr_target*1000 < end_time and target_idx < len(audio_targets):
            curr_idx = time_to_frames(curr_target, sr=SR, hop_length=HOP_LEN)
#            print(f"start idx {start_idx} start time {start_time} end time {end_time} curr target {curr_target} curr idx {curr_idx}")
            targets[curr_idx-start_idx-1] = 1 #subtract 1 since difference tells you the distance, not the index (off by one error)
            target_idx += 1
            if target_idx >= len(audio_targets):
                break

#            if prev_idx == curr_idx: #NOTE: test
#                print(f"prev is {prev*1000}, curr is {curr_target*1000}")
#            prev = curr_target
#            prev_idx = curr_idx

            curr_target = audio_targets[target_idx] / 1000
        return np.array(targets)

    def slice_windows(self, audio_feat, start_idx, max_len):
        '''
        retrieve <self.max_seq_len> many 15x80x3 (centered input frame+context, mel frequency, window length) slices of <audio_feat>, starting from <start_idx>
        '''
        slices = []
        i = 0
        while i < max_len:
            idx = start_idx+i
            s = get_frames_at_idx(audio_feat, idx)
            slices.append(s)
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
    
    def find_first(self, target, lst):
        '''
        return index of first element in lst larger or equal to <target>
        '''
        p1 = 0
        p2 = len(lst)-1

        while p1 != p2-1:
            curr = (p1+p2) // 2
            val = lst[curr]
            if val == target:
                return curr
            elif val < target:
                p1 = curr
            elif val > target:
                p2 = curr
        if lst[p1] >= target:
            return p1
        return p2

    def bin_search_closest(self, target, lst, lenience=5):
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
                    diff = grp.attrs.get("diff_name")
                    audio_feat = grp_set.get("audio_feat")[:]
                    audio_targets = grp.get("audio_targets")[:]
                    osu_tokens = grp.get("osu_tokens")[:]
                    deltas_fwd = grp.get("deltas_fwd")[:]
                    deltas_back = grp.get("deltas_back")[:]
                    stars = grp.get("stars")[()]
                    aim = grp.get("aim")[()]
                    speed = grp.get("speed")[()]
                    sample = {
                            "song": curr_name,
                            "diff": diff,
                            "audio_feat": audio_feat,
                            "audio_targets": audio_targets,
                            "aim": aim,
                            "deltas_fwd": deltas_fwd,
                            "deltas_back": deltas_back,
                            "osu_tokens": osu_tokens,
                            "stars": stars,
                            "speed": speed
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
