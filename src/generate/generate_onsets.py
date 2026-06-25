from generate.onset_generator_class import OnsetGenerator 
from models.onset.onset_model import OnsetModel
from models.onset.onset_config import OnsetConfig
from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent

if __name__ == '__main__':
    print("initializing model...")
    model = OnsetModel(OnsetConfig())
    model_state_name = ""
    model_state_path = BASE_DIR.parent.parent / "onset_checkpoints" / model_state_name
    print("loading weights into model...")
    checkpoint = torch.load(model_state_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("creating generator...")
    generator = OnsetGenerator(model)

    print("beginning onset generation...")
    #TODO: take file name from cmd line arguments
    path = "" #TODO
    onsets = generator.song_to_onsets(path)

    plot_path = "" #TODO
    generator.plot_thresholds(onsets, plot_path)
