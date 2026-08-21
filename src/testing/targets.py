import h5py
import random
import testing.helpers as th
from dataclass.obg_audio_dataset import OBGAudioDataset
from pathlib import Path
from configs.audio_config import AudioConfig
from librosa import frames_to_time

BASE_DIR = Path(__file__).parent
h5path = BASE_DIR / 'datasets' / 'test'
configA = AudioConfig()
full_data = OBGAudioDataset(
    h5path=h5path,
    max_seq_len=configA.sequence_len,
    benchmark=True
)

test_data = OBGAudioDataset(
    h5path=h5path,
    max_seq_len=configA.sequence_len,
    test=True
)

def full_targets_count(idx):
    print("idx is", idx)
    inputs, difficulty, targets = full_data[idx]
    total_targets = targets.sum()
    osu_targets = th.calc_targets(idx)
    print(f"total targets: {total_targets}, targets from osu file: {osu_targets}")

def test_targets_count(idx):
    print("idx is", idx)
    inputs, difficulty, targets, start_idx = test_data[idx]
    total_targets = targets.sum()

    start_time = frames_to_time(start_idx, sr=configA.sr, hop_length=configA.hop_len).item()
    start_time = start_time * 1000
    osu_targets = th.calc_targets(idx, start_time=start_time, max_len=configA.sequence_len)
    print(f"total targets: {total_targets}, targets from osu file: {osu_targets}")


if __name__ == '__main__':
    with h5py.File(h5path, 'r') as h:
        n_samples = h.attrs.get("num_samples")
        idx = random.randrange(0, n_samples)
    idx = 1
#    full_targets_count(idx)
    test_targets_count(idx)
