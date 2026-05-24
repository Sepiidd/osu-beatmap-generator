from pathlib import Path
import os
from tokens.tokenizer import obg_tokenizer
from preprocess.converter import obj_converter

BASE_DIR = Path(__file__).parent
tokenizer = obg_tokenizer()
converter = obj_converter(tokenizer=tokenizer)

def process_osu(osu_path):
    with open(osu_path, "r") as osz:
        line = ""
        while not line.startswith("[HitObjects]"):
            line = osz.readline()
        line = osz.readlines()
        output = converter.hitobject_seq_to_tok(line)
        print("output length is:", len(output))
        print("first and last tokens are:", output[0], output[-1])


if __name__ == "__main__":
    song_name = "2114540 METALITY UNITED - Saint Catastrophe"
    diff_name = "METALITY UNITED - Saint Catastrophe (Booliix) [Blessings of the Divine Above].osu"
    path = BASE_DIR.parent.parent.parent / "data" / song_name / diff_name
    process_osu(path)
