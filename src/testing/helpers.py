from pathlib import Path
from preprocess.hitobject_utils import get_hitobjects 
from preprocess.hitobject_utils import find_type

BASE_DIR = Path(__file__).parent
OSZ_DIR = BASE_DIR / 'data' / 'osz'

def calc_targets(idx):
    targets = 0
    circles = 0
    slider_heads = 0
    n_slides = 0
    spinners = 0

    hitobjects = get_hitobjects(OSZ_DIR / f"{idx}.osu")
    print("number of hitobjects:", len(hitobjects))
    for hitobject in hitobjects:
        hitobject = hitobject.strip()
        t, slides = find_type(hitobject)
        if t == "circle":
            circles += 1
        if t.startswith("slider"):
            n_slides += slides
            slider_heads += 1
        if t.startswith("spinner"):
            spinners += 1
    targets = circles + slider_heads + n_slides
    print(f"circles: {circles}, slider_heads: {slider_heads}, slides: {n_slides}, spinners (not included in target count): {spinners}")
    return targets
