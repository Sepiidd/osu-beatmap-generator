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
from configs.audio_config import AudioConfig
from configs.preprocess_config import PreprocessConfig 
from preprocess.hitobject_utils import augment_reflect_x
from preprocess.hitobject_utils import augment_reflect_y
from preprocess.hitobject_utils import augment_reflect_xy
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

def in_h5(path, song_id, diff_name):
    with h5py.File(path, 'a', track_order=True) as f:
        if f"{song_id}" in f: #skip sample if already exists
            sg = f.get(f"{song_id}")
            diff_lst = [sg.get(g).attrs.get("diff_name", None) for g in list(sg.keys())]
            if diff_name in diff_lst:
                print("already exists:", song_id, "|", diff_name)
                return True
    return False

def process_osu(osz):
    line = ""
    while not line.startswith("AudioFilename: "):
        line = osz.readline()
    mp3 = line.strip()[15:]
    while not line.startswith("[HitObjects]"):
        line = osz.readline()
    hitobj_idx = osz.tell()
    line = osz.readlines()
    tokens = converter.hitobject_seq_to_tok(line)
    ms_seq, forward_deltas, backward_deltas = converter.hitobject_seq_to_ms(line)
    return (np.array(tokens), np.array(ms_seq), np.array(forward_deltas), np.array(backward_deltas), mp3, hitobj_idx)

def process_audio(audio, sr):
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
        deltas_back
    ):
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

        #important metadata
        if num_diffs == 0:
            set_grp.attrs["start_range"] = num_samples
        set_grp.attrs["end_range"] = num_samples
        set_grp.attrs["diffs"] = num_diffs + 1
        f.attrs["num_samples"] = num_samples + 1

def augment_osu(osz, hitobj_idx):
    augmented_x_flip_raw = augment_reflect_x(osz) 
    augmented_x_flip = converter.hitobject_seq_to_tok(augmented_x_flip_raw) 
    osz.seek(hitobj_idx) #reset file pointer to start of hitobjects

    augmented_y_flip_raw = augment_reflect_y(osz)
    augmented_y_flip = converter.hitobject_seq_to_tok(augmented_y_flip) 
    osz.seek(hitobj_idx) #reset file pointer to start of hitobjects

    augmented_xy_flip = [] #dummy for now (disk space limit)
    augmented_xy_flip_raw = augment_reflect_xy(osz)
    augmented_xy_flip = converter.hitobject_seq_to_tok(augmented_xy_flip)

    osz.seek(hitobj_idx) #reset file pointer to start of hitobjects
    return (augmented_x_flip, augmented_y_flip, augmented_xy_flip)

def augment_audio(song_path, features targets, fwd, bwd):
    #rate change
    augmented_targets = targets[:]
    augmented_fwd = fwd[:]
    augmented_bwd = bwd[:] 
    augmented_speed = augment_speed(song_path, augmented_targets, augmented_fwd, augmented_bwd)

    #pitch change
    augmented_pitch = augment_pitch(song_path)

    #frequency mask
    augmented_freq = augment_frequency_mask(features)

    return (augmented_speed, augmented_targets, augmented_fwd, augmented_bwd, augmented_pitch, augmented_freq)

def process_one(data_path, h5path, song_name, diff_name):
    song_id = song_name.split(" ")[0]
    exists_original = False
    exists_speedup = False
    exists_pitchup = False
    exists_mask = False
    if in_h5(h5path, song_id, diff_name):
        exists_original = True
    if in_h5(h5path, song_id+"-x_flip_speedup", "x_flip_speedup:"+diff_name):
        exists_speedup = True 
    if in_h5(h5path, song_id+"-y_flip_pitchup", "y_flip_pitchup:"+diff_name):
        exists_pitchup = True
    if in_h5(h5path, song_id+"-xy_flip_freq_mask", "xy_flip_freq_mask:"+diff_name):
        exists_mask = True 

    #osu
    osu_path = data_path / song_name / diff_name
    with open(osu_path, "r") as osz:
        osu, ms_seq, forward_deltas, backward_deltas, mp3, hitobj_idx = process_osu(osz)
        osz.seek(hitobj_idx) #reset file pointer to start of hitobjects

        #augmentation (osu)
        augmented_x_flip, augmented_y_flip, augmented_xy_flip, = augment_osu(osz, hitobj_idx)

    #audio
    song_path = data_path / song_name / mp3
    audio, sr = load(path=song_path, sr=SR)
    features = process_audio(audio, sr)

    #augmentation (audio)
    augmented_speed, augmented_targets, augmented_fwd, augmented_bwd, augmented_pitch, augmented_freq = augment_audio(song_path, features, ms_seq, forward_deltas, backward_deltas)
    speed_features = process_audio(augmented_speed, sr)

    to_save = []
    if not exists_original: to_save.append((song_id, diff_name, features, ms_seq, osu, forward_deltas, backward_deltas))
    if not exists_speedup: to_save.append((song_id+"-x_flip_speedup", "x_flip_speedup:"+diff_name, speed_features, augmented_targets, augmented_x_flip, augmented_fwd, augmented_bwd))
    if not exists_pitchup: to_save.append((song_id+"-y_flip_pitchup", "y_flip_pitchup:"+diff_name, features, ms_seq, augmented_y_flip, forward_deltas, backward_deltas))
    if not exists_mask: to_save.append(song_id+"-xy_flip_freq_mask", "xy_flip_freq_mask:"+diff_name, features, ms_seq, augmented_xy_flip, forward_deltas, backward_deltas))

    return to_save

def save_set(arg_list, h5path):
    for args in arg_list:
        save_point(h5path, *args)

def process_many(data_path, h5path):
    for d in data_path.iterdir():
        song_name = d.name
        
        augmentations = []
        originals = []
        speedups = []
        pitch = []
        masked = []
        augmentations.append(originals)
        augmentations.append(speedups)
        augmentations.append(pitch)
        augmentations.append(masked)

        for f in d.iterdir():
            if f.suffix == ".osu":
                diff_name = f.name
                to_save = process_one(data_path, h5path, song_name, diff_name)
                if len(to_save)>=1: originals.append(to_save[0])
                if len(to_save)>=2: speedups.append(to_save[1])
                if len(to_save)>=3: pitch.append(to_save[2])
                if len(to_save)>=4: masked.append(to_save[3])
        #save entire mapsets at once
        for augmentation in augmentations:
            save_set(augmentation, h5path)
    return

if __name__ == "__main__":
    spin = input("1 for train, 2 for validation, 3 for test:\n")
    split = "train" if spin == '1' else ("validation" if spin == '2' else ("test" if spin == '3' else "default"))
    print("split is", split)

    h5path = Path(configP.h5_parent) / "datasets" / split
    data_dir = Path(configP.data_parent) / "data" / split 
    process_many(data_dir, h5path)
