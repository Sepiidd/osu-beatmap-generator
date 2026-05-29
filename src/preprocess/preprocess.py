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

def process_osu_hitobjects(osu_path):
    with open(osu_path, "r") as osz:
        line = ""
        while not line.startswith("[HitObjects]"):
            line = osz.readline()
        line = osz.readlines()
        tokens = converter.hitobject_seq_to_tok(line)
        ms_seq, forward_deltas, backward_deltas = converter.hitobject_seq_to_ms(line)
    return (tokens, ms_seq, forward_deltas, backward_deltas, line)

def process_audio(audio_path):
    audio, sr = load(path=audio_path, sr=SR)
    features = []
    for i, n in enumerate(N_FFT):
        transformed = stft(audio, n_fft=n, hop_length=HOP_LEN, center=True)
        transformed = np.abs(transformed) ** 2 #square of magnitudes
        mel_filter = mel(sr=SR, n_fft=n, n_mels=N_MEL)
        filtered = mel_filter @ transformed
        log_scaled = power_to_db(filtered, ref=np.max)
        normalized = (log_scaled - np.mean(log_scaled)) / np.std(log_scaled)

        features.append(normalized)
    features = np.stack(features, axis=-1)
    return features

def save_onset_point(audio_features, targets):
    #TODO
    pass

def save_object_point(audio_features, deltas_back, deltas_fwd, token_targets):
    #TODO
    pass    

if __name__ == "__main__":
    song_name = "2114540 METALITY UNITED - Saint Catastrophe"
    diff_name = "METALITY UNITED - Saint Catastrophe (Booliix) [Blessings of the Divine Above].osu"
    path = BASE_DIR.parent.parent.parent / "data" / song_name / diff_name

    osu, ms_seq, forward_deltas, backward_deltas, lines = process_osu_hitobjects(path)

    audio_path = BASE_DIR.parent.parent.parent / "data" / song_name / "audio.mp3"
    features = process_audio(audio_path)

    #NOTE: instead of splicing here, save entire processed spectrogram along with entire ms_list and splice during batch retrieval
    #onset_frames, onset_targets = gen_onset_points(features, lines)

    print("lines mem size (bytes) is", np.array(lines).nbytes)
    print("osu tokens mem size (bytes) is", np.array(osu).nbytes, "with shape", np.array(osu).shape)
    print("ms_seq targets mem size (bytes) is", np.array(ms_seq).nbytes, "with shape", np.array(ms_seq).shape)
    print("forward deltas mem size (bytes) is", np.array(forward_deltas).nbytes, "with shape", np.array(forward_deltas).shape)
    print("backward deltas mem size (bytes) is", np.array(backward_deltas).nbytes, "with shape", np.array(forward_deltas).shape)
    print("audio features mem size (bytes) is", features.nbytes, "with shape", features.shape)
    total_bytes = 0
    total_bytes += np.array(osu).nbytes
    total_bytes += np.array(lines).nbytes 
    total_bytes += np.array(ms_seq).nbytes 
    total_bytes += np.array(forward_deltas).nbytes 
    total_bytes += np.array(backward_deltas).nbytes 
    total_bytes += features.nbytes
    print("TOTAL MEM USAGE (BYTES):", total_bytes)

    print("")
    print("ms_seq first 6:", ms_seq[:6])
    print("forward_deltas first 5:", forward_deltas[:5])
    print("backward_deltas first 5:", backward_deltas[:5])
