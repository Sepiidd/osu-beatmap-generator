import h5py
from pathlib import Path

def read_group(f, indent=0):
    ls = list(f.keys())
    for k in ls:
        thing = f.get(k)
        vert = "L" if indent else ""
        hori = "_"*(indent-1) if indent else ""
        attri = " "*indent if indent else ""
        attrp = attri + k + " attributes are:\n" if len(thing.attrs.items()) > 0 else ""
        print(attrp, end="")
        for key, val in thing.attrs.items():
            print(attri + f"{key}: {val}")
        if isinstance(thing, h5py.Group):
            print(vert + hori + k, "is a group:")
            read_group(thing, indent=indent+2)
        elif isinstance(thing, h5py.Dataset):
            if thing.shape == ():
                print(vert + hori + k, "is a dataset with size", thing.shape, "it's contents are", thing[()])
            else:
                print(vert + hori + k, "is a dataset with size", thing.shape)
    return

if __name__ == "__main__":
#    split = 'train'
#    split = 'validation'
    split = 'test'
#    path = Path(__file__).parent.parent / "datasets" / split 
    path = Path(__file__).parent / "testing" / "datasets" /split 
    with h5py.File(path, "r") as f:
        print("f attrs are:")
        for key, val in f.attrs.items():
            print(f"{key}: {val}")
        print()
        read_group(f)
