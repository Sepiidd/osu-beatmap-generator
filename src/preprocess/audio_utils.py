from librosa import time_to_frames
import numpy as np

def get_frames_at_time(features, time_ms, max_frame_idx):
    time_s = time_ms / 1000
    frame_idx = time_to_frames(time_s)
    frames = None
    if frame_idx-7 < 0: #frame_idx too close to start
        num_pad = abs(frame_idx-7)
        frames = features[:, frame_idx:frame_idx+8, :]
        frames = np.pad(frames, ((0,0), (num_pad, 0), (0,0)), mode="constant")
    elif (frame_idx+7)-max_frame_idx > 0: #frame_idx too close to end
        num_pad = abs((frame_idx+7)-max_frame_idx)
        frames = features[:, frame_idx-7:frame_idx+1, :]
        frames = np.pad(frames, ((0,0), (0, num_pad), (0,0)), mode="constant")
    else:
        frames = features[:, frame_idx-7:frame_idx+8, :]
    return frames
