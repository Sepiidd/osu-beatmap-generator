import time
import sys
import numpy as np
import h5py
from math import ceil
from pathlib import Path
from librosa import load, power_to_db, stft
from librosa.filters import mel
from librosa import time_to_frames   

from tokens.tokenizer import obg_tokenizer
from preprocess.converter import obj_converter
from preprocess.audio_utils import get_frames_at_time

#globals
BASE_DIR = Path(__file__).parent
N_FFT=[512, 1024, 2048] #n_fft param in stft (also sets win_len with no argument)
HOP_LEN=221 #stft stride param, ~10ms=HOP_LEN/22050 * 1000 where 22050 is default librosa sr
N_MEL=80
SR=22050

tokenizer = obg_tokenizer(load_tokens=False)
converter = obj_converter(tokenizer=tokenizer)

def process_osu(osu_path):
    with open(osu_path, "r") as osz:
        line = ""
        while not line.startswith("AudioFilename: "):
            line = osz.readline()
        mp3 = line.strip()[15:]
        while not line.startswith("[HitObjects]"):
            line = osz.readline()
        line = osz.readlines()
        tokens = converter.hitobject_seq_to_tok(line)
        ms_seq, forward_deltas, backward_deltas = converter.hitobject_seq_to_ms(line)
    return (np.array(tokens), np.array(ms_seq), np.array(forward_deltas), np.array(backward_deltas), mp3)

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

def save_point(
        path, song_id, diff_name, 
        audio_features, 
        audio_targets,
        osu_tokens,
        deltas_fwd,
        deltas_back
    ):
    with h5py.File(path, 'a', track_order=True) as f:
        if f"{song_id}" in f: #skip sample if already exists
            sg = f.get(f"{song_id}")
            diff_lst = [sg.get(g).attrs.get("diff_name", None) for g in list(sg.keys())]
            if diff_name in diff_lst:
                print("already exists:", song_id, "|", diff_name)
                return
        #hierarchy creation
        print("saving:", song_id, "|", diff_name)
        set_grp = f.require_group(f"{song_id}")
        num_diffs = int(set_grp.attrs.get("diffs", 0))
        num_samples = int(f.attrs.get("num_samples", 0))
        grp = set_grp.create_group(f"{num_samples}")
        grp.attrs["diff_name"] = diff_name
        
        #store data
        if "audio_feat" not in set_grp:
            set_grp.create_dataset("audio_feat", data=audio_features)
        grp.create_dataset("audio_targets", data=audio_targets)
        grp.create_dataset("osu_tokens", data=osu_tokens)
        grp.create_dataset("deltas_fwd", data=deltas_fwd)
        grp.create_dataset("deltas_back", data=deltas_back)

        #important metadata
        if num_diffs == 0:
            set_grp.attrs["start_range"] = num_samples
        set_grp.attrs["end_range"] = num_samples
        set_grp.attrs["diffs"] = num_diffs + 1
        f.attrs["num_samples"] = num_samples + 1

def process_one(data_path, h5path, song_name, diff_name):
    osu, ms_seq, forward_deltas, backward_deltas, mp3 = process_osu(data_path / song_name / diff_name)
    features = process_audio(data_path / song_name / mp3)

    song_id = song_name.split(" ")[0]
    save_point(h5path, song_id, diff_name, features, ms_seq, osu, forward_deltas, backward_deltas)

def process_many(data_path, h5path):
    for d in data_path.iterdir():
        song_name = d.name
        for f in d.iterdir():
            if f.suffix == ".osu":
                diff_name = f.name
                process_one(data_path, h5path, song_name, diff_name)
    return

if __name__ == "__main__":
    h5path = BASE_DIR.parent.parent / "datasets" / "partition0"
    data_dir = BASE_DIR.parent.parent / "data"
    process_many(data_dir, h5path)
