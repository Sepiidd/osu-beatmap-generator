import librosa
import math
import shutil
import torch
from pathlib import Path
from preprocess.hitobject_utils import find_time

BASE_DIR = Path(__file__).parent

class HitsoundGenerator():
    def __init__(self, onset_generator, osu_songs_dir):
        self.onset_generator = onset_generator
        self.osu_songs_dir = osu_songs_dir

    def generate_hitsound_map(self, audio_dir, audio_filename, difficulty_args, output_dir, features=None):
        '''
        <difficulty_args>: dict containing <HPDrainRate>, <CircleSize>, <OverallDifficulty>, <ApproachRate>, <SliderMultiplier>, <SliderTickRate>
        '''
        osu_file_content = 'osu file format v14\n\n'
        general = self.generate_general(audio_filename)
        editor = self.generate_editor()
        metadata = self.generate_metadata()
        difficulty = self.generate_difficulty(**difficulty_args)
        events = self.generate_events()
        colours = self.generate_colours()
        timestamps, _ = self.onset_generator.path_to_onsets(audio_dir / audio_filename) if features is not None else self.onset_generator.song_to_onsets(features)
        hitobjects = self.generate_hitobjects(timestamps)
        timing_points = self.generate_timing_points(audio_dir / audio_filename, timestamps[0])

        osu_file_content += general+'\n'
#        osu_file_content += editor+'\n'
        osu_file_content += metadata+'\n'
        osu_file_content += difficulty+'\n'
#        osu_file_content += events+'\n'
        osu_file_content += timing_points+'\n'
#        osu_file_content += colours+'\n'
        osu_file_content += hitobjects

        #save to file with .osu extension 
        self.save_to_osu_file(output_dir, audio_dir, audio_filename, osu_file_content)

    def save_to_osu_file(self, output_dir, audio_dir, audio_filename, osu_content):
        dirpath = Path(output_dir).resolve()
        audiodir = Path(audio_dir).resolve()

        osupath = dirpath / "test - test (test) [test].osu"
        audiopath = dirpath / audio_filename
        shutil.copy(audiodir / audio_filename, audiopath)
        with open(osupath, 'w') as f:
            f.write(osu_content)

    def generate_general(self, AudioFilename):
        general='[General]\n'
        general+='AudioFilename: '+AudioFilename+'\n'
        general+='Countdown: 0'+'\n'
        return general

    def generate_editor(self):
        editor='[Editor]\n'
        return editor 

    def generate_metadata(self):
        metadata='[Metadata]\n'
        metadata+='Title:test\n'
        metadata+='Artist:test\n'
        metadata+='Version:test\n'
        return metadata

    def generate_difficulty(self, HPDrainRate, CircleSize, OverallDifficulty, ApproachRate, SliderMultiplier, SliderTickRate):
        difficulty='[Difficulty]\n'
        difficulty+='HPDrainRate:'+HPDrainRate+'\n'
        difficulty+='CircleSize:'+CircleSize+'\n'
        difficulty+='OverallDifficulty:'+OverallDifficulty+'\n'
        difficulty+='ApproachRate:'+ApproachRate+'\n'
        difficulty+='SliderMultiplier:'+SliderMultiplier+'\n'
        difficulty+='SliderTickRate:'+SliderTickRate+'\n'
        return difficulty 

    def generate_events(self):
        events='[Events]\n'
        return events 

    def generate_timing_points(self, audio_path, first_timestamp):
        #TODO: 
        timing_points='[TimingPoints]\n'

        #default, song-wide bpm
        audio, sr = librosa.load(audio_path)
        bpm = math.ceil(librosa.beat.tempo(y=audio, sr=sr)[0]*2)

        beat_len = str((1 / bpm) * 1000 * 60)
        meter='4'
        sample_set='0'
        sample_idx='0'
        vol='100'
        uninherited='1'
        effects='0'

        time = int(first_timestamp.item())
        t_point = str(time) + ',' + beat_len + ',' + meter + ',' + sample_set + ',' + sample_idx + ',' + vol + ',' + uninherited + ',' + effects
        timing_points += t_point + '\n'

        return timing_points 

    def generate_colours(self):
        colours='[Colours]\n'
        return colours 

    def generate_hitobjects(self, timestamps):
        hitobjects='[HitObjects]\n'

        x = '256'
        y = '192'
        t = '1'
        hs = '0'
        prev_t = -1
        for idx, time in enumerate(timestamps): 
            time = int(time.item())
            if time == prev_t:
                continue
            hitobject = x + ',' + y + ',' + str(time) + ',' + t + ',' + hs + ',0:0:0:0:'
            if idx != len(timestamps)-1:
                hitobject += '\n'
                
            hitobjects+=hitobject
            prev_t = time
        return hitobjects 

if __name__ == '__main__':
    pass
