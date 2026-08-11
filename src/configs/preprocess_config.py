from dataclasses import dataclass, field

@dataclass
class PreprocessConfig:
    data_parent: str = "/home/sebastian/projects/osu-beatmap-gen/osu-beatmap-generator"
    h5_parent: str = "/home/sebastian/projects/osu-beatmap-gen/osu-beatmap-generator"
#    checkpoint: str = "/media/sebastian/My Passport/Sebastian/obg-data"
    checkpoint: str = "/home/sebastian/projects/osu-beatmap-gen/osu-beatmap-generator/"
