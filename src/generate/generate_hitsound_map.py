import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from configs.gen_config import GenConfig
from configs.training_config import TrainingConfig
from configs.onset_config import OnsetConfig
from dataclass.obg_audio_dataset import OBGAudioDataset
from models.onset.onset_model import OnsetModel
from generate.onset_generator_class import OnsetGenerator
from generate.hitsound_map_generator import HitsoundGenerator
from pathlib import Path
from training.helpers_train import make_generator


BASE_DIR = Path(__file__).parent


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = "1100diff_film"
    model_path = BASE_DIR.parent.parent / 'onset_saved' / model_name
    model_config = OnsetConfig()
    model = OnsetModel(model_config)

    #load model params
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    #create onset generator and hitsound map generator
    generator = OnsetGenerator(model)
    osu_songs_dir = Path('~/.local/share/osu-wine/osu!/Songs')
    osu_songs_dir = osu_songs_dir.expanduser().resolve()
    hitsound_gen = HitsoundGenerator(generator, osu_songs_dir)

    #create dataloader
    config = TrainingConfig()
    train_set = OBGAudioDataset(config.train_path, config.sequence_len, benchmark=True)
    val_set = OBGAudioDataset(config.validation_path, config.sequence_len, benchmark=True)

    #get specific input, targets
#    test_idx = 0
    test_idx = 4
#    test_idx = 357
    inputs, difficulty, targets = train_set[test_idx]

    #produce predictions
#    audio_filename = "we-are-dreamers.mp3"
    audio_filename = "saint_catastrophe.mp3"
#    audio_filename = "s-heaven.mp3"
    audio_input_dir = BASE_DIR.parent.parent / 'music' 
    output_dir = osu_songs_dir / 'test'
    difficulty_args = {
        "HPDrainRate": '1',
        "CircleSize": '4',
        "OverallDifficulty": '10',
        "ApproachRate": '9.6',
        "SliderMultiplier": '2',
        "SliderTickRate": '1'
    }
    hitsound_gen.generate_hitsound_map(audio_input_dir, audio_filename, difficulty_args, output_dir, features=inputs, map_difficulty=difficulty)

    



