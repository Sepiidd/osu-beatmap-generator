import librosa.effects as E
import librosa
import numpy as np
import torchaudio.transforms as T
import torch.nn.functional as F
import torch
import math
from librosa import time_to_frames
from configs.audio_config import AudioConfig

configA = AudioConfig()

def get_frames_at_time(features, time_ms):
    time_s = time_ms / 1000
    frame_idx = time_to_frames(time_s)
    return get_frames_at_idx(features, frame_idx)

def get_frames_at_idx(audio_feat, idx, context_len=7):
    max_frame = audio_feat.shape[1]-1
    frames = None
    if idx-context_len < 0: #idx too close to start
        num_pad = abs(idx-context_len)
        frames = audio_feat[:, idx-(context_len-num_pad):idx+context_len+1, :]
        frames = np.pad(frames, ((0,0), (num_pad, 0), (0,0)), mode="constant")
    elif (idx+context_len)-max_frame > 0: #idx too close to end
        num_pad = abs((idx+context_len)-max_frame)
        frames = audio_feat[:, idx-context_len:max_frame+1, :]
        frames = np.pad(frames, ((0,0), (0, num_pad), (0,0)), mode="constant")
    else:
        frames = audio_feat[:, idx-context_len:idx+context_len+1, :]
    return frames

def augment_pitch(audio, shift=2):
    """
    increase/decrease pitch by <shift> steps (default 2) 

    <audio>: raw audio path 
    """
    audio, sr = librosa.load(audio)
    return E.pitch_shift(audio, sr=configA.sr, n_steps=shift)

def augment_speed(audio, target_ms, rate=1.2):
    """
    increase speed of audio by <rate> (default 1.2x), update related osu information accordingly

    <audio>: raw audio path 
    <target_ms>: array of timestamps representing ground truth onset
    <rate>: multiplier which to speed up/down the audio by
    """
    audio, sr = librosa.load(audio)
    augmented_a = E.time_stretch(audio, rate=rate)
   
    #update osu related timings
    i = 0
    while i < len(target_ms):
        target_ms[i] = target_ms[i] / rate
        i+=1
    return augmented_a, target_ms

def augment_speed_all(audio, target_ms, fwd_d, bwd_d, rate=1.2):
    """
    increase speed of audio by <rate> (default 1.2x), update related osu information accordingly

    <audio>: raw audio path 
    <target_ms>: array of timestamps representing ground truth onset
    <fwd_d>: difference in time between onset <i> and <i+1> for all <i> in <target_ms>
    <bwd_d>: difference in time between onset <i> and <i-1> for all <i> in <target_ms>
    <rate>: multiplier which to speed up/down the audio by
    """
    audio, sr = librosa.load(audio)
    augmented_a = E.time_stretch(audio, rate=rate)
   
    #update osu related timings
    i = 0
    while i < len(target_ms):
        target_ms[i] = target_ms[i] / rate
        fwd_d[i] = fwd_d[i] / rate
        bwd_d[i] = bwd_d[i] / rate
        i+=1
    return augmented_a

def augment_frequency_mask(audio_feats, max_aug=5):
    """
    zero out a random frequency band

    <audio_feats>: np array of shape (80, N, 3) where N is the number of frames for the original audio series
    """
    augmented = audio_feats.copy()
    num_augmentations = np.random.randint(max_aug)
    for i in range(num_augmentations):
        band_selection = np.random.randint(configA.n_mel)
        augmented[band_selection, :, :] = 0
    return augmented 

def augment_temporal_mask(audio_feats, max_aug=30):
    """
    zero out a random time

    <audio_feats>: np array of shape (80, N, 3) where N is the number of frames for the original audio series
    """
    augmented = audio_feats.copy()
    num_augmentations = np.random.randint(max_aug)
    for i in range(num_augmentations):
        num_frames = audio_feats.shape[1]
        band_selection = np.random.randint(num_frames-num_augmentations)
        augmented[:, band_selection:band_selection+num_augmentations, :] = 0
    return augmented 

def apply_hamming_window(to_smooth, ham_len, device='cpu'):
    #apply hamming window across batch
    ham_window = torch.hamming_window(ham_len, periodic=False).to(device)

    #padding to maintain <output_len>=<input_len>
    #normalize hamming window to sum to one, keeps output
    smoothed = F.conv1d(to_smooth.view(1, 1, -1), ham_window.view(1, 1, -1) / ham_window.sum(), padding=ham_len//2)
    return smoothed

def nms_binary_search(lst, target, tiebreaker):
    """
    returns index of target, in case of duplicates uses <tiebreaker> index

    <target>: single value tensor (avoid using .item() in case of numerical stability)
    <tiebreaker>: integer representing the index in the original predictions tensor
    """
    p1 = 0
    p2 = len(lst)-1
    while p1<=p2:
        mid = (p1+p2)//2
        v = lst[mid][1]
        if torch.equal(v, target):
            #check for duplicates (VERY IMPORTANT!!)
            l_bounded = False
            r_bounded = False
            dist = 1
            while not l_bounded and not r_bounded:
                l = mid-dist
                r = mid+dist
                lv = None if l < 0 else lst[l][1]
                rv = None if r > len(lst)-1 else lst[r][1]
                lidx = None if l < 0 else lst[l][0]             #index of value in original predictions tensor
                ridx = None if r > len(lst)-1 else lst[r][0]    #index of value in original predictions tensor

                if lv is None or not torch.equal(lv, v):
                    l_bounded = True
                elif lidx == tiebreaker: #lv is not none and torch.equal, use tiebreaker
                    return l

                if rv is None or not torch.equal(rv, v):
                    r_bounded = True
                elif ridx == tiebreaker: #rv is not none and torch.equal, use tiebreaker
                    return r
                dist+=1
            #no duplicates, return mid
            return mid
        elif torch.gt(v, target): #v greater than target
            p2=mid-1
        elif torch.lt(v, target): #v less than target
            p1=mid+1
        num+=1
    return None #force some error downstream, element should be in lst

def apply_nms(predictions, pred_threshold, hop_len, sr, bpm=None):
    '''
    applies non maximum suppression via scoring, enforces a distance of the equivalent of a 1/16th beat between maximums

    returns the indices of predictions after performing nms
    '''
    p_bool = (predictions > pred_threshold).squeeze()

    #calculate minimum distance between peaks
    time_per_beat = (60000/bpm)/16 if bpm else 25 #time in ms per 1/16th beat
    time_per_frame = hop_len/sr * 1000 #time in ms per frame
    frame_per_beat = time_per_beat/time_per_frame
    r = math.floor(0.9*frame_per_beat) 
    
    #zip predictions and respective indices for sorting
    zipped = list(enumerate(predictions)) #form of [(idx, probability),...]
    zipped_f = [z for z in zipped if z[1]>pred_threshold]
    zipped_s = sorted(zipped_f, key=lambda z: z[1]) #sort by probability

    while len(zipped_s) > 0:
        idx, confidence = zipped_s.pop(0)
        start = 0 if idx-r < 0 else idx-r
        end = len(predictions)-1 if idx+r >= len(predictions) else idx+r


        for i in range(start, end+1):
            if not p_bool[i] or i==idx:
                continue
            p_bool[i] = False
            #remove tuple (i, predictions[i]) from  zipped_s
            i_idx = nms_binary_search(zipped_s, predictions[i], i)
            zipped_s.pop(i_idx)
    #convert p_bool into indices of true onsets
    predictions_idx = torch.nonzero(p_bool, as_tuple=True)[0]
    return predictions_idx

