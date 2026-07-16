import librosa.effects as E
import librosa
import numpy as np
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

def augment_speed(audio, target_ms, fwd_d, bwd_d, rate=1.2):
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

def augment_frequency_mask(audio_feats):
    """
    zero out a random frequency band

    <audio_feats>: np array of shape (80, N, 3) where N is the number of frames for the original audio series
    """
    band_selection = np.random.randint(configA.n_mel)
    audio_feats[band_selection, :, :] = 0
    return audio_feats









