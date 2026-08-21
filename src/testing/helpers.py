from pathlib import Path
from preprocess.hitobject_utils import get_hitobjects 
from preprocess.hitobject_utils import find_type
from preprocess.hitobject_utils import find_time
from configs.audio_config import AudioConfig

BASE_DIR = Path(__file__).parent
OSZ_DIR = BASE_DIR / 'data' / 'osz'
configA = AudioConfig()

def calc_targets(idx, start_time=0, max_len=None):
    #TODO: does NOT count slides that are happening at the start time, and might miss slider starting at the end (see saint catastrophe, start index 12884)
    targets = 0
    circles = 0
    slider_heads = 0
    n_slides = 0
    spinners = 0

    hitobjects = get_hitobjects(OSZ_DIR / f"{idx}.osu")
    ms_per_frame = (1000/configA.sr) * configA.hop_len
    end_time = start_time + max_len*ms_per_frame +ms_per_frame/2 if max_len is not None else None
    start_time = start_time - ms_per_frame/2
    print("in helper, start and end time are", start_time, end_time)
    print("number of hitobjects:", len(hitobjects))
    for i, hitobject in enumerate(hitobjects):
        hitobject = hitobject.strip()
        time = find_time(hitobject)
        if time < start_time:
            continue
        if end_time and time > end_time:
            break

        t, slides = find_type(hitobject)
        if t == "circle":
            circles += 1
        if t.startswith("slider"):
            n_slides += slides
            slider_heads += 1
        if t.startswith("spinner"):
            spinners += 1
    targets = circles + slider_heads + n_slides + spinners
    print(f"circles: {circles}, slider_heads: {slider_heads}, slides: {n_slides}, spinners: {spinners}")
    return targets

