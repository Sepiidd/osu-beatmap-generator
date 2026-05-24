def find_type(obj):
    separated = obj.strip().split(",")
    t = int(separated[3])
    b_idx = first_bit_idx_int(t)
    match b_idx:
        case 0:
            return ("circle", 0)
        case 1:
            curve_type = find_slider_type(obj)
            slides = find_num_slides(obj)
            return (curve_type, slides)
        case 2:
            return ("spinner", 0)

def find_slider_type(obj):
    separated = obj.strip().split(",")
    pipe_sep = separated[5].split("|")
    s_type = pipe_sep[0]
    match s_type.lower():
        case "b":
            return "slider-b"
        case "l":
            return "slider-l"
        case "p":
            return "slider-p"
        case "c":
            return "slider-c"

def find_num_slides(obj):
    separated = obj.strip().split(",")
    return separated[6]

def first_bit_idx_int(num):
    for i in range(8):
        is_set = (num >> i) & 1
        if is_set:
            return i
    return -1
