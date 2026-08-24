import time
import sys
import numpy as np
import h5py
import rosu_pp_py as R
from math import ceil
from pathlib import Path
from librosa import load, power_to_db, stft
from librosa.filters import mel
from librosa import time_to_frames   
from tokens.tokenizer import obg_tokenizer
from preprocess.converter import obj_converter
from configs.audio_config import AudioConfig
from configs.preprocess_config import PreprocessConfig 
from preprocess.hitobject_utils import augment_reflect_x
from preprocess.hitobject_utils import augment_reflect_y
from preprocess.hitobject_utils import augment_reflect_xy
from preprocess.hitobject_utils import tpoint_time_uninherited 
from preprocess.audio_utils import augment_pitch 
from preprocess.audio_utils import augment_speed 
from preprocess.audio_utils import augment_frequency_mask

#globals
BASE_DIR = Path(__file__).parent
configA = AudioConfig()
configP = PreprocessConfig()

N_FFT = configA.n_fft
HOP_LEN = configA.hop_len
N_MEL = configA.n_mel
SR = configA.sr

tokenizer = obg_tokenizer(load_tokens=False)
converter = obj_converter(tokenizer=tokenizer)
diff_calculator = R.Difficulty(lazer=False)

def in_h5(path, song_id, diff_name):
    with h5py.File(path, 'a', track_order=True) as f:
        if f"{song_id}" in f: #skip sample if already exists
            sg = f.get(f"{song_id}")
            diff_lst = [sg.get(g).attrs.get("diff_name", None) for g in list(sg.keys())]
            if diff_name in diff_lst:
                print("already exists:", song_id, "|", diff_name)
                return True
    return False

def set_in_h5(path, song_id):
    with h5py.File(path, 'a', track_order=True) as f:
        if f"{song_id}" in f: #skip set if already exists 
            print("song set already exists, skipping all diffs:", song_id)
            return True
    return False

def process_osu(osz):
    line = ""
    while not line.startswith("AudioFilename:"):
        line = osz.readline()
    end_idx = line.find(':')
    mp3 = line.strip()[end_idx+1:]
    mp3 = mp3.strip()

    while not line.startswith("SliderMultiplier:"):
        line = osz.readline()
    end_idx = line.find(':')
    slider_mult = line.strip()[end_idx+1:]
    slider_mult = float(slider_mult.strip())

    while not line.startswith("[TimingPoints]"):
        line = osz.readline()
    line = osz.readline() #first timing point
    
    uninherited_tpoints = [] #updated list, earliest timing at start  
    inherited_tpoints = [] #updated list, earliest timing at start
    while not line.startswith("\n"):
        time, uninherited = tpoint_time_uninherited(line)
        if uninherited:
            uninherited_tpoints.append((time, line.strip()))
        else:
            inherited_tpoints.append((time, line.strip()))
        line = osz.readline()

#    print("uninherited", len(uninherited_tpoints))
#    print("inherited is", len(inherited_tpoints))

    while not line.startswith("[HitObjects]"):
        line = osz.readline()

    hitobj_idx = osz.tell()
    line = osz.readlines()
    tokens = converter.hitobject_seq_to_tok(line)
    ms_seq, forward_deltas, backward_deltas = converter.hitobject_seq_to_ms(line, slider_mult, uninherited_tpoints, inherited_tpoints)
    return (np.array(tokens), np.array(ms_seq), np.array(forward_deltas), np.array(backward_deltas), mp3, hitobj_idx)

def process_audio(audio, sr):
    features = []
    for i, n in enumerate(N_FFT):
        transformed = stft(audio, n_fft=n, hop_length=HOP_LEN, center=True)
        transformed = np.abs(transformed) ** 2 #square of magnitudes
        mel_filter = mel(sr=SR, n_fft=n, n_mels=N_MEL)
        filtered = mel_filter @ transformed
        log_scaled = power_to_db(filtered, ref=np.max)
        normalized = (log_scaled - np.mean(log_scaled, axis=1, keepdims=True)) / np.std(log_scaled, axis=1, keepdims=True)

        features.append(normalized)
    features = np.stack(features, axis=-1)
    #shape of features is: 80 13212 3
#    m, t, w = features.shape
#    print("shape of features is:", m, t, w)
    return features

def save_point(
        path, song_id, diff_name, 
        audio_features, 
        audio_targets,
        osu_tokens,
        deltas_fwd,
        deltas_back,
        stars,
        aim,
        speed
    ):
    #TODO: reset state when exception or keyboard interrupt occurred here
    with h5py.File(path, 'a', track_order=True) as f:
        #hierarchy creation
        print("saving:", song_id, "|", diff_name)
        set_grp = f.require_group(f"{song_id}")
        num_diffs = int(set_grp.attrs.get("diffs", 0))
        num_samples = int(f.attrs.get("num_samples", 0))
        num_songs = int(f.attrs.get("num_songs", 0))
        grp = set_grp.create_group(f"{num_samples}")
        grp.attrs["diff_name"] = diff_name
        
        #store data
        if "audio_feat" not in set_grp:
            set_grp.create_dataset("audio_feat", data=audio_features)
            f.attrs["num_songs"] = num_songs + 1
        grp.create_dataset("audio_targets", data=audio_targets)
        grp.create_dataset("osu_tokens", data=osu_tokens)
        grp.create_dataset("deltas_fwd", data=deltas_fwd)
        grp.create_dataset("deltas_back", data=deltas_back)
        grp.create_dataset("stars", data=stars)
        grp.create_dataset("aim", data=aim)
        grp.create_dataset("speed", data=speed)

        #important metadata
        if num_diffs == 0:
            set_grp.attrs["start_range"] = num_samples
        set_grp.attrs["end_range"] = num_samples
        set_grp.attrs["diffs"] = num_diffs + 1
        f.attrs["num_samples"] = num_samples + 1

def process_one(data_path, h5path, song_name, diff_name):
    song_id = song_name.split(" ")[0]
    if in_h5(h5path, song_id, diff_name):
       return [] 

    #osu
    osu_path = data_path / song_name / diff_name
    with open(osu_path, "r") as osz:
        osu, ms_seq, forward_deltas, backward_deltas, mp3, hitobj_idx = process_osu(osz)
        osz.seek(hitobj_idx) #reset file pointer to start of hitobjects

    #osu difficulty calc    
    bmap = R.Beatmap(path = str(osu_path))
    ratings = diff_calculator.calculate(bmap)
    stars = ratings.stars
    aim = ratings.aim
    speed = ratings.speed

    #audio
    #NOTE: pitch/time-shift audio augmentation must be performed here (no way to delay augmentation to data loader stage)
    song_path = data_path / song_name / mp3
    audio, sr = load(path=song_path, sr=SR)
    features = process_audio(audio, sr)

    to_save = []
    to_save.append((song_id, diff_name, features, ms_seq, osu, forward_deltas, backward_deltas, stars, aim, speed))

    return to_save

def save_set(arg_list, h5path):
    for args in arg_list:
        save_point(h5path, *args)

def process_many(data_path, h5path):
    for d in data_path.iterdir():
        song_name = d.name
        song_id = song_name.split(" ")[0]
        if set_in_h5(h5path, song_id): #catch duplicate map sets, will mess with indexing
            continue

        to_save = []
        for f in d.iterdir():
            if f.suffix == ".osu":
                diff_name = f.name
                to_save.extend(process_one(data_path, h5path, song_name, diff_name))

        #save entire mapsets at once
        for song_args in to_save:
            save_point(h5path, *song_args)
    return

if __name__ == "__main__":
    spin = input("1 for train, 2 for validation, 3 for test:\n")
    split = "train" if spin == '1' else ("validation" if spin == '2' else ("test" if spin == '3' else "default"))
    print("split is", split)

    h5path = Path(configP.h5_parent) / "datasets" / split
    data_dir = Path(configP.data_parent) / "data" / split 
    process_many(data_dir, h5path)
