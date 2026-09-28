from preprocess.preprocess import process_audio
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
from librosa import load
from configs.audio_config import AudioConfig
from configs.training_config import TrainingConfig
from configs.gen_config import GenConfig 
from configs.onset_config import OnsetConfig 
from preprocess.audio_utils import get_frames_at_idx
from preprocess.audio_utils import apply_hamming_window
from preprocess.audio_utils import apply_nms

#globals
configA = AudioConfig()
configT = TrainingConfig()
configG = GenConfig()
configO = OnsetConfig()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OnsetGenerator():
    def __init__(self, model):
        self.model = model
        self.batch_size = configG.batch_size
        self.overlap_len = configG.overlap_len 
        self.sequence_len = configA.sequence_len

        self.prediction_threshold = configG.prediction_threshold
        self.hamming_window_len = configG.hamming_window_len

        self.model.eval()

    def batch_to_onsets(self, inputs, difficulties):
        '''
        converts a batch of input data into onset probabilities

        <difficulties>: tensor containing [stars, aim, speed]
        '''
        inp = torch.from_numpy(inputs)
        inp = inp.to(device)
        diff = difficulties.to(device)
        with torch.no_grad():
            logits = self.model(inp, diff)
        #apply sigmoid for binary classification probabilities
        return F.sigmoid(logits) 

    def make_batch_diff(self, diff, batch_size):
        '''
        clone <diff> to batch size <batch_size>
        '''
        if batch_size == 1: #final batch NOTE: bandaid fix kind of solution tbh
            new = diff[0:1, :]
        else:
            new = diff.expand(batch_size, -1)
        return new

    def make_batch_sequence(self, features, start, end):
        '''
        create a batch of size <self.batch_size>, containing sequences of length <self.sequence_len>
        lowers batch size if necessary
        '''

        if hasattr(features, "is_cuda") and features.is_cuda:
            features = features.to('cpu') #copy to cpu for numpy if required

        m, t, w = features.shape
        len_batch = self.sequence_len * self.batch_size
        batch_seq = []
        for i in range(start, end): 
            if i>=t: #indexing past features max length
                break
            splice = get_frames_at_idx(features, i) #(15, 80, 3) shape
            batch_seq.append(splice)
        batch_seq = np.stack(batch_seq)

        if t-start < len_batch:
            batch_seq = np.reshape(batch_seq, (1, -1, 15, configA.n_mel, len(configA.n_fft)))
        else:
            batch_seq = np.reshape(batch_seq, (self.batch_size, self.sequence_len, 15, configA.n_mel, len(configA.n_fft)))
        return batch_seq

    def path_to_onsets(self, path, difficulty=None):
        '''
        convert entire song found at <path> into a list of onset probabilities
        '''
        audio, sr = load(path=path, sr=configA.sr)
        features = process_audio(audio, sr)
        return self.song_to_onsets(features, difficulty)

    def song_to_onsets(self, features, difficulty=None):
        '''
        convert features spectrogram to list of onset probabilities
        NOTE: expects input in form of (m, t, w) as described below
        '''
        if difficulty is None:
            stars_default = 7.0
            aim = stars_default/2
            speed = stars_default/2
            difficulty = torch.tensor([stars_default, aim, speed], dtype=torch.float32)

        m, t, w = features.shape #shape of: mel bins, time (frame idx), window length

        num_predictions = torch.zeros(t).to(device)
        predictions = torch.zeros(t).to(device)

        len_batch = self.sequence_len * self.batch_size
        idx = 0
        iter_num = 0
        #loop over features and track index:
        while idx < t:
            #batch, splice, index features
            batch_seq = self.make_batch_sequence(features, idx, idx+len_batch) #reshaped to (t, m, w)
            bsize = batch_seq.shape[0]
            difficulty = self.make_batch_diff(difficulty, bsize)

            #perform model predictions
            batch_predictions = self.batch_to_onsets(batch_seq, difficulty)
            batch_predictions = batch_predictions.view(-1) #flatten batch into one sequence again

            predictions_len = batch_predictions.shape[0]
            batch_num_p = torch.ones(predictions_len).to(device)

            batch_num_p = F.pad(batch_num_p, (idx, t-(idx+predictions_len)))
            batch_predictions = F.pad(batch_predictions, (idx, t-(idx+predictions_len)))

            predictions = predictions + batch_predictions
            num_predictions = num_predictions + batch_num_p 

            print(f"completed iter num {iter_num}, making predictions over indices ({idx}-{idx+len_batch}). NOTE: t={t}, pred len is {predictions_len}")

            idx += len_batch-self.overlap_len
            iter_num += 1
        #average predictions
        predictions = predictions / num_predictions 
        smoothed = apply_hamming_window(predictions, self.hamming_window_len, device)
        smoothed = smoothed.squeeze()
        self.plot_thresholds(predictions, "plot_test", label="raw")
        self.plot_thresholds(smoothed.squeeze(), "plot_test", label="smoothed")

        #convert positive prediction indices into timestamps
        predictions_bool = (smoothed > self.prediction_threshold).squeeze() #remove extra 1 dimensions along with boolean filter
        predictions_idx = torch.nonzero(predictions_bool, as_tuple=True)[0] 
        
        #apply nms and get indices of true onset predictions
        #TODO: get bpm info from somewhere
        print("applying nms...")
        predictions_idx = apply_nms(smoothed, self.prediction_threshold, configA.hop_len, configA.sr, bpm=None)

        times = predictions_idx * configA.hop_len / configA.sr #calculation described by <https://librosa.org/doc/latest/generated/librosa.frames_to_time.html>
        times = times * 1000 #time in ms
        return times, predictions

    def plot_thresholds(self, probabilities, file_name, timestamps=[], alpha=0.7, start_plot_from=10000, plot_first_many=150, label=""):
        '''
        plots predictions' onset probabilities on a line graph, save to <file_name>.png
        '''
        prob = probabilities.cpu().detach().numpy()


        if start_plot_from is None and plot_first_many is None:
            plt.plot(prob, alpha=alpha, label=label)
            crossed = np.where(prob > self.prediction_threshold)[0]
        else:
            plt.plot(prob[start_plot_from:start_plot_from+plot_first_many], alpha=alpha, label=label)
            crossed = np.where((prob[start_plot_from:start_plot_from+plot_first_many]) > self.prediction_threshold)[0]

        if label == "smoothed" and start_plot_from and plot_first_many:
            plt.vlines(x=crossed, ymin=0, ymax=1, color='green', alpha=0.5, linewidth=1, label='predictions')
#            print(f"{len(crossed)} frames above {self.prediction_threshold}: {crossed}")
        plt.axhline(y=self.prediction_threshold, color='r', linestyle='-')

        plt.xlabel(f"First {plot_first_many} Indices From Index {start_plot_from}")
#        if plot_first_many:
#            plt.xticks(np.arange(0, plot_first_many, 10))
        plt.ylabel("Onset Probability")
        plt.ylim(0,1)
        plt.yticks(np.arange(0, 1, 0.1))
        plt.title("Onset Probability Over Time")
        plt.legend()
        plt.savefig(file_name + ".png")
