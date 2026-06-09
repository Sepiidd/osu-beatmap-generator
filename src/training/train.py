from dataclass.obg_audio_dataset import OBGAudioDataset
from pathlib import Path
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).parent
SEQUENCE_LEN = 512

if __name__ == "__main__":
    h5path = BASE_DIR.parent.parent / "datasets" / "partition0"
    dataset = OBGAudioDataset(h5path, SEQUENCE_LEN)
    dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True
            )
    for inputs, targets in dataloader:
        print("inputs has shape", inputs.shape, "with size (bytes)", inputs.nbytes)
        print("targets has shape", targets.shape, "with size (bytes)", inputs.nbytes)
        break;
        #TODO: training stuff goes here
