from generate.onset_generator_class import OnsetGenerator 
from models.onset.onset_model import OnsetModel
from configs.onset_config import OnsetConfig
from pathlib import Path
import torch
import sys

BASE_DIR = Path(__file__).parent

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("please pass <song_name> from ../../songs/<song_name>")
        exit(1)
    arguments = sys.argv[1:]

    print("initializing model...")
    model = OnsetModel(OnsetConfig())
    model_state_name = "test_680iters_10_maps"
    model_state_path = BASE_DIR.parent.parent / "onset_saved" / model_state_name
    print("loading weights into model...")
    checkpoint = torch.load(model_state_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("creating generator...")
    generator = OnsetGenerator(model)

    print("beginning onset generation...")
    print("command line arguments are:", arguments)
    path = BASE_DIR.parent.parent / 'music' / arguments[0]
    onsets = generator.song_to_onsets(path)
    print("shape of onsets is", onsets.shape)
    print("onsets list is", onsets)
