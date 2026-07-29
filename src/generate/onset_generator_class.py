from preprocess.preprocess import process_audio
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from librosa import load
from configs.audio_config import AudioConfig
from configs.training_config import TrainingConfig
from configs.gen_config import GenConfig 
from preprocess.audio_utils import get_frames_at_idx

#globals
configA = AudioConfig()
configT = TrainingConfig()
configG = GenConfig()

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

    def batch_to_onsets(self, inputs):
        '''
        converts a batch of input data into onset probabilities, along with post model prediction operations e.g. hamming window
        '''
        inp = torch.from_numpy(inputs)
        inp = inp.to(device)
        with torch.no_grad():
            logits = self.model(inp)
        #apply sigmoid for binary classification probabilities
        return F.sigmoid(logits) 

    def make_batch_sequence(self, features, start, end):
        '''
        create a batch of size <self.batch_size>, containing sequences of length <self.sequence_len>
        lowers batch size if necessary
        '''
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

    def path_to_onsets(self, path):
        '''
        convert entire song found at <path> into a list of onset probabilities
        '''
        audio, sr = load(path=path, sr=configA.sr)
        features = process_audio(audio, sr)
        return self.song_to_onsets(features)

    def song_to_onsets(self, features):
        '''
        convert features spectrogram to list of onset probabilities
        NOTE: expects input in form of (m, t, w) as describedc directly below
        '''
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

            #perform model predictions
            batch_predictions = self.batch_to_onsets(batch_seq)
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

        #plot for testing
#        print("shape of predictions before hamming", predictions.shape)
#        self.plot_thresholds(predictions, "plot_test")

        #apply hamming window across batch
        ham_window = torch.hamming_window(self.hamming_window_len, periodic=False).to(device)

        #padding to maintain <output_len>=<input_len>
        #normalize hamming window to sum to one, keeps output  
        smoothed = F.conv1d(predictions.view(1, 1, -1), ham_window.view(1, 1, -1) / ham_window.sum(), padding=self.hamming_window_len//2) 
        self.plot_thresholds(smoothed.squeeze(), "plot_test_smoothed")

        #convert positive prediction indices into timestamps
        predictions_bool = (smoothed > self.prediction_threshold).squeeze() #remove extra 1 dimensions along with boolean filter
#        print("predictions bool shape is", predictions_bool.shape)
        predictions_idx = torch.nonzero(predictions_bool, as_tuple=True)[0] 
#        print("predictions idx are", predictions_idx)
        
        times = predictions_idx * configA.hop_len / configA.sr #calculation described by <https://librosa.org/doc/latest/generated/librosa.frames_to_time.html>
        return times, predictions

    def plot_thresholds(self, probabilities, file_name):
        '''
        plots predictions' onset probabilities on a line graph, save to <file_name>.png
        '''
        plt.plot(probabilities.cpu().detach().numpy())
        plt.xlabel("Ordered Index")
        plt.ylabel("Onset Probability")
        plt.ylim(0,1)
        plt.title("Onset Probability Over Time")
        plt.savefig(file_name + ".png")
