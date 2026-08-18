import h5py
import random
from dataclass.obg_audio_dataset import OBGAudioDataset
from pathlib import Path
from configs.audio_config import AudioConfig
import testing.helpers as th

BASE_DIR = Path(__file__).parent
h5path = BASE_DIR / 'datasets' / 'test'
configA = AudioConfig()
testing_data = OBGAudioDataset(
    h5path=h5path,
    max_seq_len=configA.sequence_len,
    benchmark=True
)

def check_targets_count(idx):
    print("idx is", idx)
    inputs, difficulty, targets = testing_data[idx]
    total_targets = targets.sum()
    osu_targets = th.calc_targets(idx)
    print(f"total targets: {total_targets}, targets from osu file: {osu_targets}")


if __name__ == '__main__':
    with h5py.File(h5path, 'r') as h:
        n_samples = h.attrs.get("num_samples")
        idx = random.randrange(0, n_samples)
    check_targets_count(idx)
