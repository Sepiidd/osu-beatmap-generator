from dataclasses import dataclass, field

@dataclass
class PreprocessConfig:
    data_parent: str = "/home/sebastian/projects/osu-beatmap-gen/osu-beatmap-generator"
    h5_parent: str = "/home/sebastian/projects/osu-beatmap-gen/osu-beatmap-generator"
    checkpoint: str = "/home/sebastian/projects/osu-beatmap-gen/osu-beatmap-generator/"

#    data_parent: str = "/home/sebastian/projects/osu-beatmap-gen/osu-beatmap-generator/src/testing" #NOTE: testing
#    h5_parent: str = "/home/sebastian/projects/osu-beatmap-gen/osu-beatmap-generator/src/testing" #NOTE: testing
