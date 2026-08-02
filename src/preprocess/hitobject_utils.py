#globals
X_MIDDLE = 256 #playfield center on x-axis
Y_MIDDLE = 192 #playfield center on y-axis

def find_time(obj):
    obj = obj.strip()
    separated = obj.split(',')
    return separated[2]

def find_type(obj):
    separated = obj.strip().split(",")
    t = int(separated[3])
    b_idx = first_bit_idx_int(t)
    match b_idx:
        case 0:
            return ("circle", 0) #0 slides
        case 1:
            curve_type = find_slider_type(obj)
            slides = find_num_slides(obj)
            return (curve_type, slides)
        case 2:
            return ("spinner", 0) #0 slides

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

def x_flip_cl(obj):
    stripped = obj.strip()
    split = stripped.split(',')
    x = int(split[0])
    to_mid = dist_to_mid_x(x)
    new_x = X_MIDDLE + to_mid 

    end_idx = stripped.find(',')
    new_str = str(new_x) + stripped[end_idx:] 


    return new_str

def x_flip_anchor(obj):
    stripped = obj.strip()
    start_idx = stripped.find('|')
    split1 = stripped[start_idx+1:]
    anchors_raw = split1[:split1.find(',')]
    anchors = anchors_raw.split('|')
    
    new_anchors = ''
    for anchor in anchors:
        a_split = anchor.split(':')
        x = int(a_split[0])
        y = int(a_split[1])

        to_mid = dist_to_mid_x(x)
        new_x = X_MIDDLE + to_mid 

        new_anchor = str(new_x) + ':' + str(y)
        new_anchors = new_anchors + new_anchor if new_anchors == '' else new_anchors + '|' + new_anchor
    
    new_str = stripped[:start_idx+1] + new_anchors + split1[split1.find(','):]
    return new_str

def y_flip_cl(obj):
    stripped = obj.strip()
    split = stripped.split(',')
    x = int(split[0])
    y = int(split[1])
    to_mid = dist_to_mid_y(int(y))
    new_y = Y_MIDDLE + to_mid 

    end_idx = stripped.find(',', stripped.find(',') + 1)
    new_str = str(x) + ',' + str(new_y) + stripped[end_idx:] 

    return new_str

def y_flip_anchor(obj):
    stripped = obj.strip()
    start_idx = stripped.find('|')
    split1 = stripped[start_idx+1:]
    anchors_raw = split1[:split1.find(',')]
    anchors = anchors_raw.split('|')
    
    new_anchors = ''
    for anchor in anchors:
        a_split = anchor.split(':')
        x = int(a_split[0])
        y = int(a_split[1])

        to_mid = dist_to_mid_y(y)
        new_y = Y_MIDDLE + to_mid 

        new_anchor = str(x) + ':' + str(new_y)
        new_anchors = new_anchors + new_anchor if new_anchors == '' else new_anchors + '|' + new_anchor
    
    new_str = stripped[:start_idx+1] + new_anchors + split1[split1.find(','):]
    return new_str

def xy_flip_cl(obj):
    stripped = obj.strip()
    split = stripped.split(',')
    x = int(split[0])
    y = int(split[1])

    to_mid = dist_to_mid_x(x)
    new_x = X_MIDDLE + to_mid 

    to_mid = dist_to_mid_y(y)
    new_y = Y_MIDDLE + to_mid 

    end_idx = stripped.find(',', stripped.find(',')+1)
    new_str = str(new_x) + ',' + str(new_y) + stripped[end_idx:] 

    return new_str

def xy_flip_anchor(obj):
    stripped = obj.strip()
    start_idx = stripped.find('|')
    split1 = stripped[start_idx+1:]
    anchors_raw = split1[:split1.find(',')]
    anchors = anchors_raw.split('|')
    
    new_anchors = ''
    for anchor in anchors:
        a_split = anchor.split(':')
        x = int(a_split[0])
        y = int(a_split[1])

        to_mid = dist_to_mid_x(x)
        new_x = X_MIDDLE + to_mid 

        to_mid = dist_to_mid_y(y)
        new_y = Y_MIDDLE + to_mid 

        new_anchor = str(new_x) + ':' + str(new_y)
        new_anchors = new_anchors + new_anchor if new_anchors == '' else new_anchors + '|' + new_anchor
    
    new_str = stripped[:start_idx+1] + new_anchors + split1[split1.find(','):]
    return new_str

def augment_reflect_x(osu):
    """
    flip all hitobjects across the x axis

    <osu>: file object referencing actual .osu file (starts at hitobjects)

    returns list of raw string hitobjects with augmentation applied
    """
    augmented = []
    line = osu.readline()
    while line:
        stripped = line.strip()
        t, slides = find_type(stripped)
        
        new_str = x_flip_cl(stripped)
        if t.startswith("slider"):
            new_str = x_flip_anchor(new_str)

        augmented.append(new_str)
        line = osu.readline()
    return augmented

def augment_reflect_y(osu):
    """
    flip all hitobjects across the y axis

    <osu>: file object referencing actual .osu file (starts at hitobjects)

    returns list of raw string hitobjects with augmentation applied
    """
    augmented = []
    line = osu.readline()
    while line:
        stripped = line.strip()
        t, slides = find_type(stripped)

        new_str = y_flip_cl(stripped)
        if t.startswith("slider"):
            new_str = y_flip_anchor(new_str)

        augmented.append(new_str)
        line = osu.readline()
    return augmented

def augment_reflect_xy(osu):
    """
    flip all hitobjects across both axes

    <osu>: file object referencing actual .osu file (starts at hitobjects)

    returns list of raw string hitobjects with augmentation applied
    """
    augmented = []
    line = osu.readline()
    while line:
        stripped = line.strip()
        t, slides = find_type(stripped)

        new_str = xy_flip_cl(stripped)
        if t.startswith("slider"):
            new_str = xy_flip_anchor(new_str)

        augmented.append(new_str)
        line = osu.readline()
    return augmented




if __name__ == "__main__":
    pass
