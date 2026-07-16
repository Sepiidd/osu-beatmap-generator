#globals
X_MIDDLE = 256 #playfield center on x-axis
Y_MIDDLE = 192 #playfield center on y-axis

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

def dist_to_mid_x(x):
    return X_MIDDLE - x

def dist_to_mid_y(y):
    return Y_MIDDLE - y

def augment_reflect_x(osu):
    """
    flip all hitobjects across the x axis

    <osu>: file object referencing actual .osu file (starts at hitobjects)

    returns list of raw string hitobjects with augmentation applied
    """
    #TODO: debug, test
    augmented = []
    line = osu.readline()
    while line:
        stripped = line.strip()
        split = stripped.split(',')
        x = split[0]
        to_mid = dist_to_mid_x(int(x))
        new_x = X_MIDDLE + to_mid if x < X_MIDDLE else X_MIDDLE - to_mid

        end_idx = stripped.find(',')
        new_str = str(new_x) + stripped[end_idx+1] 

        augmented.append(new_str)
        line = osu.readline()
    return augmented

def augment_reflect_y(osu):
    """
    flip all hitobjects across the y axis

    <osu>: file object referencing actual .osu file (starts at hitobjects)

    returns list of raw string hitobjects with augmentation applied
    """
    #TODO: debug, test
    augmented = []
    line = osu.readline()
    while line:
        stripped = line.strip()
        split = stripped.split(',')
        x = split[0]
        y = split[1]
        to_mid = dist_to_mid_y(int(y))
        new_y = Y_MIDDLE + to_mid if y < Y_MIDDLE else Y_MIDDLE - to_mid

        end_idx = stripped.find(',', stripped.find(','))
        new_str = str(x) + ',' + str(new_y) + stripped[end_idx] 

        augmented.append(new_str)
        line = osu.readline()
    return augmented

def augment_reflect_xy(osu):
    """
    flip all hitobjects across both axes

    <osu>: file object referencing actual .osu file (starts at hitobjects)

    returns list of raw string hitobjects with augmentation applied
    """
    #TODO: debug, test
    augmented = []
    line = osu.readline()
    while line:
        stripped = line.strip()
        split = stripped.split(',')
        x = split[0]
        y = split[1]

        to_mid = dist_to_mid_x(int(x))
        new_x = X_MIDDLE + to_mid if x < X_MIDDLE else X_MIDDLE - to_mid

        to_mid = dist_to_mid_y(int(y))
        new_y = Y_MIDDLE + to_mid if y < Y_MIDDLE else Y_MIDDLE - to_mid

        end_idx = stripped.find(',', stripped.find(','))
        new_str = str(new_x) + ',' + str(new_y) + stripped[end_idx] 

        augmented.append(new_str)
        line = osu.readline()
    return augmented





