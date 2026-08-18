import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAveragePrecision
from configs.gen_config import GenConfig
from configs.training_config import TrainingConfig 
from configs.onset_config import OnsetConfig 
from dataclass.obg_audio_dataset import OBGAudioDataset
from models.onset.onset_model import OnsetModel
from generate.onset_generator_class import OnsetGenerator
from pathlib import Path
from training.helpers_train import make_generator

configG = GenConfig()
configT = TrainingConfig()
BASE_DIR = Path(__file__).parent

@torch.no_grad()
def eval_loss_peakpick(generator, config):
    ctx = config.ctx
    eval_iters = config.eval_iters
    criterion = config.criterion
    device = config.device

    model.eval()
    train_set = OBGAudioDataset(config.train_path, config.sequence_len, benchmark=True)
    train_loader_full = DataLoader(
        train_set,
        batch_size=1,
        shuffle=config.shuffle,
        pin_memory=config.pin_memory,
    )

    val_set = OBGAudioDataset(config.validation_path, config.sequence_len, benchmark=True)
    validation_loader_full = DataLoader(
        val_set,
        batch_size=1,
        shuffle=config.shuffle,
        pin_memory=config.pin_memory,
    )

    train_gen = make_generator(train_loader_full, device)
    val_gen = make_generator(validation_loader_full, device)

    stats = {}
    out = {}
#    for split in ["train", "val"]:
    for split in ["train"]:
        loader = train_gen if split == 'train' else val_gen 

        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0
        aucpr_calc = BinaryAveragePrecision()

        i = 0
        while True:
#            if i >= eval_iters:
            if i >= 1:
                break

#            inputs, targets = next(loader)
            s_idx = 0
            inputs, difficulty, targets = train_set[s_idx] #specific map
            print("inputs, difficulty, targets have shapes", inputs.shape, difficulty.shape, targets.shape)
            inputs = inputs.squeeze()
            times, predictions = generator.song_to_onsets(inputs, difficulty=difficulty) 
            generator.plot_thresholds(targets.squeeze(), "plot_test_smoothed")
            print("for s_idx", s_idx, "length of times (# of onset positive predictions) is", times.shape, "number of real onsets is", targets.sum())

            #apply hamming window across batch
            ham_window = torch.hamming_window(configG.hamming_window_len, periodic=False).to(device)

            #padding to maintain <output_len>=<input_len>
            #normalize hamming window to sum to one, keeps output
            smoothed = F.conv1d(predictions.view(1, 1, -1), ham_window.view(1, 1, -1) / ham_window.sum(), padding=configG.hamming_window_len//2)

            smoothed = smoothed.squeeze()
            targets = targets.squeeze().to(device)

            aucpr_calc.update(smoothed, targets.to(int))

            predictions = (smoothed > configG.prediction_threshold).squeeze()

            batch_tp = (predictions * targets).sum()
            tp += batch_tp
            batch_fp = (predictions * (1-targets)).sum()
            fp += batch_fp
            batch_fn = ((~predictions) * targets).sum()
            fn += batch_fn

            batch_correct = (predictions == targets).sum().item()
            correct += batch_correct

            if len(targets.shape) == 1:
                total += targets.shape[0] # sequence length
            else:
                total += targets.shape[0]*targets.shape[1] #batch size * sequence length
            i+=1

        #determine stats 
        precision = (tp / (tp+fp + 1e-7)).item() #of all positive predictions, which were correct
        recall = (tp / (tp+fn + 1e-7)).item()
        f_score = (2*tp) / (2*tp+fp+fn + 1e-7)

        aucpr = aucpr_calc.compute()
        aucpr_calc.reset()
        print(f"================{split} split: f-score {f_score} aucpr {aucpr} precision {precision} recall {recall}================")

if __name__ == '__main__':
    model_name = "1000iternewset"
    model_path = BASE_DIR.parent.parent / 'onset_saved' / model_name
    model_config = OnsetConfig()
    model = OnsetModel(model_config)

    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(configT.device)

    generator = OnsetGenerator(model)

    print("running benchmark for model with name", model_name)
    eval_loss_peakpick(generator, configT)
