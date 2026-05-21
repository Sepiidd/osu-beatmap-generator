def find_type(obj):
    separated = obj.strip().split(",")
    t = separated[2]
    b_idx = first_bit_idx_int(t)
    match b_idx:
        case 0:
            return ("circle", 0)
        case 1:
            curve_type = find_slider_type(obj)
            slides = find_num_slides(separated)
            return (curve_type, slides)

def find_slider_type(obj):
    #NOTE: will cause failures in future on C type sliders prob
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

def find_num_slides(obj):
    separated = obj.strip().split(",")
    return separated[6]

def first_bit_idx_int(num):
    for i in range(8):
        is_set = (num >> i) & 1
        if is_set:
            return i
    return -1
