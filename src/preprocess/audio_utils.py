from librosa import time_to_frames
import numpy as np

def get_frames_at_time(features, time_ms):
    time_s = time_ms / 1000
    frame_idx = time_to_frames(time_s)
    return get_frames_at_idx(features, frame_idx)

def get_frames_at_idx(audio_feat, idx, context_len=7):
    max_frame = audio_feat.shape[1]-1
    frames = None
    if idx-context_len < 0: #idx too close to start
        num_pad = abs(idx-context_len)
        frames = audio_feat[:, idx:idx+context_len+1, :]
        frames = np.pad(frames, ((0,0), (num_pad, 0), (0,0)), mode="constant")
    elif (idx+context_len)-max_frame > 0: #idx too close to end
        num_pad = abs((idx+context_len)-max_frame)
        frames = audio_feat[:, idx-context_len:max_frame+1, :]
        frames = np.pad(frames, ((0,0), (0, num_pad), (0,0)), mode="constant")
    else:
        frames = audio_feat[:, idx-context_len:idx+context_len+1, :]
    return frames
