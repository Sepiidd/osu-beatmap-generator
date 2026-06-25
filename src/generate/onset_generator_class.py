from preprocess.preprocess import process_audio
import torch.nn.functional as F
import torch
import matplotlib

class OnsetGenerator():
    def __init__(self, model):
        #TODO
        self.model = model
        self.batch_size = 8 #NOTE: move to a config?
        self.overlap_len = 64 #NOTE: play around with this number
        self.sequence_len = 512

    def batch_to_onsets(self, inputs):
        '''
        converts a batch of input data into onset probabilities, along with post model prediction operations e.g. hamming window
        '''
        logits = self.model(inputs)
        #apply sigmoid for binary classification probabilities
        return F.sigmoid(logits) 


    def song_to_onsets(self, path):
        '''
        convert entire song found at <path> into a list of onset probabilities
        '''
        #TODO: overlap windows, sum probabilities and overlap count, then average predictions?
        #TODO
        features = torch.from_numpy(process_audio(path)) #entire song -> filtered spectrogram
        #loop over features and track index:
            #batch, splice, index features
            #perform model predictions
        #apply hamming window across batch
        #apply fixed threshold to obtain predictions
        #convert positive prediction indices into timestamps

        

    def plot_thresholds(self, probabilities, file_name):
        '''
        plots predictions' onset probabilities on a line graph
        '''
        #TODO
        pass
