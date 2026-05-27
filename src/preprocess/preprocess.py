import time
import sys
from math import ceil
from pathlib import Path
from librosa import load, power_to_db, stft
from librosa.filters import mel
from librosa import time_to_frames   
import numpy as np

from tokens.tokenizer import obg_tokenizer
from preprocess.converter import obj_converter
from preprocess.audio_utils import get_frames_at_time

#globals
BASE_DIR = Path(__file__).parent
N_FFT=[512, 1024, 2048] #n_fft param in stft (also sets win_len with no argument)
HOP_LEN=221 #stft stride param, ~10ms=HOP_LEN/22050 * 1000 where 22050 is default librosa sr
N_MEL=80
SR=22050

tokenizer = obg_tokenizer(load_tokens=True)
converter = obj_converter(tokenizer=tokenizer)

def gen_single_onset(features, target_hitobjects, curr_timestamp, context_len=7):
    #get timestamp frame and surrounding frames
    max_frame_idx = features.shape[1]-1
    frames = get_frames_at_time(features, curr_timestamp, max_frame_idx)
    #check timestamp of upcoming target (determines positive or negative target)
    target_timestamp = int(target_hitobjects[0].strip().split(",")[2]) if len(target_hitobjects)>0 else -100 
    target = 1 if abs(curr_timestamp-target_timestamp) <= 10 else 0
    #   if positive target, remove from array
    if target:
        target_hitobjects.pop(0)
    return (frames, target)

def gen_onset_points(features, target_hitobjects):
    onset_frames_all = []
    onset_targets = []
    first_ms = int(target_hitobjects[0].strip().split(",")[2])
    last_ms = int(target_hitobjects[-1].strip().split(",")[2])
    first_ms_floor = (first_ms // 10) * 10
    last_ms_ceil = ceil(last_ms / 10) * 10
    for ms in range(first_ms_floor, last_ms_ceil+1, 10):
        onset_frames, onset_target = gen_single_onset(features, target_hitobjects, ms)
        onset_frames_all.append(onset_frames)
        onset_targets.append(onset_target)
    onset_frames_all = np.stack(onset_frames_all, axis=0)
    onset_targets  = np.stack(onset_targets)
    return onset_frames_all, onset_targets

def process_osu_hitobjects(osu_path):
    with open(osu_path, "r") as osz:
        line = ""
        while not line.startswith("[HitObjects]"):
            line = osz.readline()
        line = osz.readlines()
        tokens = converter.hitobject_seq_to_tok(line)
    return (tokens, line)

def process_audio(audio_path):
    audio, sr = load(path=audio_path, sr=SR)
    features = []
    for i, n in enumerate(N_FFT):
        transformed = stft(audio, n_fft=n, hop_length=HOP_LEN, center=True)
        transformed = np.abs(transformed ** 2) #square of magnitudes
        mel_filter = mel(sr=SR, n_fft=n, n_mels=N_MEL)
        filtered = mel_filter @ transformed
        log_scaled = power_to_db(filtered, ref=np.max)
        normalized = (log_scaled - np.mean(log_scaled)) / np.std(log_scaled)

        features.append(log_scaled)
    features = np.stack(features, axis=-1)
    return features

if __name__ == "__main__":
    song_name = "2114540 METALITY UNITED - Saint Catastrophe"
    diff_name = "METALITY UNITED - Saint Catastrophe (Booliix) [Blessings of the Divine Above].osu"
    path = BASE_DIR.parent.parent.parent / "data" / song_name / diff_name

    osu, lines = process_osu_hitobjects(path)

    audio_path = BASE_DIR.parent.parent.parent / "data" / song_name / "audio.mp3"
    features = process_audio(audio_path)
    print("features shape is", features.shape)

    #NOTE: instead of splicing here, save entire processed spectrogram along with entire ms_list and splice during batch retrieval
    #onset_frames, onset_targets = gen_onset_points(features, lines)

    print("osu mem size (bytes) is", np.array(osu).nbytes)
    print("lines mem size (bytes) is", np.array(lines).nbytes)
    print("features mem size (bytes) is", features.nbytes)
    #print("for all onset points, frame mem size (bytes) is", onset_frames.nbytes) 
    #print("for all onset point, target mem size (bytes) is", onset_targets.nbytes) 
    total_bytes = 0
    total_bytes += np.array(osu).nbytes
    total_bytes += np.array(lines).nbytes 
    total_bytes += features.nbytes
    #total_bytes += onset_frames.nbytes
    #total_bytes += onset_targets.nbytes
    print("TOTAL MEM USAGE (BYTES):", total_bytes)
