from preprocess.preprocess import process_audio
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from configs.audio_config import AudioConfig
from configs.training_config import TrainingConfig
from preprocess.audio_utils import get_frames_at_idx

#globals
configA = AudioConfig()
configT = TrainingConfig()

class OnsetGenerator():
    def __init__(self, model):
        #TODO
        self.model = model
        self.batch_size = configT.batch_size
        self.overlap_len = 64 #NOTE: play around with this number
        self.sequence_len = configA.sequence_len

    def batch_to_onsets(self, inputs):
        '''
        converts a batch of input data into onset probabilities, along with post model prediction operations e.g. hamming window
        '''
        inp = torch.from_numpy(inputs)
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
            splice = get_frames_at_idx(features, i)
            batch_seq.append(splice)
            splice = np.swapaxes(splice, 0, 1)
        batch_seq = np.stack(batch_seq)

        if t-start < len_batch:
            batch_seq = np.reshape(batch_seq, (1, -1, 15, configA.n_mel, len(configA.n_fft)))
        else:
            batch_seq = np.reshape(batch_seq, (self.batch_size, self.sequence_len, 15, configA.n_mel, len(configA.n_fft)))
        return batch_seq

    def song_to_onsets(self, path):
        '''
        convert entire song found at <path> into a list of onset probabilities
        '''
        #TODO: overlap windows, sum probabilities and overlap count, then average predictions?
        features = process_audio(path) #entire song -> filtered spectrogram
        m, t, w = features.shape #shape of: mel bins, time (frame idx), window length

        num_predictions = torch.zeros(t)
        predictions = torch.zeros(t)

        len_batch = self.sequence_len * self.batch_size
        idx = 0
        iter_num = 0
        #loop over features and track index:
        while idx < t:
            print(f"starting iter num {iter_num}, making predictions over indices ({idx}-{idx+len_batch}). NOTE: t={t}")
            #batch, splice, index features
            batch_seq = self.make_batch_sequence(features, idx, idx+len_batch)

            #perform model predictions
            batch_predictions = self.batch_to_onsets(batch_seq)
            batch_predictions = batch_predictions.view(-1) #flatten batch into one sequence again

            predictions_len = batch_predictions.shape[0]
            print("pred len is", predictions_len)
            batch_num_p = torch.ones(predictions_len)

            batch_num_p = F.pad(batch_num_p, (idx, t-(idx+predictions_len)))
            batch_predictions = F.pad(batch_predictions, (idx, t-(idx+predictions_len)))

            predictions = predictions + batch_predictions
            num_predictions = num_predictions + batch_num_p 

            idx += len_batch
        #average predictions
        predictions = predictions / num_predictions #TODO: ensure all predictions are in range [0,1]

        #plot for testing
        self.plot_thresholds(predictions, "plot_test")

        #apply hamming window across batch
        #apply fixed threshold to obtain predictions
        #convert positive prediction indices into timestamps
        

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
