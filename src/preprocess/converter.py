from preprocess.hitobject_utils import find_type
from preprocess.hitobject_utils import find_len
from preprocess.hitobject_utils import find_end_time
from preprocess.hitobject_utils import tpoint_beatlen
from preprocess.hitobject_utils import tpoint_sv

class obj_converter():
    def __init__(self, tokenizer, special_start=["<CIRCLE>", "<SLIDER_HEAD_BEZIER>", "<SLIDER_HEAD_LINEAR>", "<SLIDER_HEAD_PERFECT>"]):
        self.special_start = special_start
        self.tokenizer = tokenizer

    def word_to_hitobject(self, obj_arr):
        '''
        convert a set of words into the corresponding hitobject
        '''
        #TODO
        obj = ""
        match obj_arr[0]:
            case "<CIRCLE>":
                pass
            case "<SLIDER_HEAD_BEZIER>":
                pass
            case "<SLIDER_HEAD_LINEAR>":
                pass
            case "<SLIDER_HEAD_PERFECT>":
                pass
        return obj

    def tok_seq_to_hitobject(self, seq):
        '''
        converts a sequence of tokens into their respective hitobjects
        '''
        #TODO
        seq_obj = []
        idx = 0
        curr = []
        while idx < len(seq):
            word = self.tokenizer.decode(seq[idx])
            if len(curr) == 0: #determine current object type (circle, slider-b/l/p)
                curr.append(word)
            elif seq[idx] not in self.special_start:
                curr.append(word)
            else:
                obj = token_to_hitobject(curr)
                seq_obj.append(obj)
                curr=[word]
        return seq_obj

    def hitobject_to_tok(self, obj):
        '''
        converts a single hitobject into an array of tokens which identify the input object
        '''
        tokens = []
        obj_type, slides = find_type(obj)
        separated = obj.strip().split(",")
        if obj_type == "circle":
            name = self.tokenizer.encode("<CIRCLE>")
            x = self.tokenizer.encode(f"X_{separated[0]}")
            y = self.tokenizer.encode(f"Y_{separated[1]}")
            tokens.extend([name, x, y])
        elif obj_type.startswith("slider"):
            name = ""
            match obj_type:
                case "slider-b":
                    name = self.tokenizer.encode("<SLIDER_HEAD_BEZIER>")
                case "slider-l":
                    name = self.tokenizer.encode("<SLIDER_HEAD_LINEAR>")
                case "slider-p":
                    name = self.tokenizer.encode("<SLIDER_HEAD_PERFECT>")
                case _: #ignore catmull sliders and everything else
                    return tokens
            x = self.tokenizer.encode(f"X_{separated[0]}")
            y = self.tokenizer.encode(f"Y_{separated[1]}")
            tokens.extend([name, x, y])
            anchor_s = separated[5].split("|")[1:]
            anchor_arr = []
            for anchor in anchor_s:
                anchor_name = self.tokenizer.encode("<ANCHOR>")
                pos = anchor.split(":")
                anchor_x = self.tokenizer.encode(f"X_{pos[0]}")
                anchor_y = self.tokenizer.encode(f"Y_{pos[1]}")
                anchor_arr.extend([anchor_name, anchor_x, anchor_y])
            tokens.extend(anchor_arr)
        elif obj_type == "spinner":
            name = self.tokenizer.encode("<SPINNER>")
            tokens.append(name)
        else:
            return tokens
        return tokens
    
    def hitobject_seq_to_tok(self, seq):
        '''
        takes a sequence of hitobjects, converts them to a list of tokens
        '''
        tokens = [self.tokenizer.encode("<BOS>")]
        for obj in seq:
            tokens.extend(self.hitobject_to_tok(obj.strip()))
        tokens.append(self.tokenizer.encode("<EOS>"))
        return tokens

    def timing_deltas(self, prev, obj, nxt, complete_time):
        ms_obj = int(obj.strip().split(",")[2])
        ms_prev = int(prev.strip().split(",")[2]) if prev else None
        ms_nxt = int(nxt.strip().split(",")[2]) if nxt else None

        t, slides = find_type(obj)

        #circle, spinner case
        if not t.startswith('slider'): 
            forward = ms_nxt - ms_obj if nxt else -1
            backward = ms_obj - ms_prev if prev else -1
            return [forward], [backward]
        
        #slider case
        fwd = []
        bwd = []
        for i in range(slides+1):
            if i == 0: #prev is a real object, next is the slider itself
                backward = ms_obj - ms_prev if prev else -1
                forward = complete_time
            elif i == slides: #prev is the slider itself, next is a real object
                backward = complete_time
                forward = ms_nxt - ms_obj if nxt else -1
            else: #both prev and next is the slider itself
                backward = complete_time
                forward = complete_time
            backward = float(backward)
            forward = float(forward)
            bwd.append(backward)
            fwd.append(forward)
        return fwd, bwd

    def hitobject_to_ms(self, obj, slider_mult, uninherited, inherited):
        u_time, u_point = uninherited[0]
        u_time = float(u_time)
        i_time, i_point = inherited[0]
        i_time = float(i_time)
        
        u_future, u_next = uninherited[1] if len(uninherited)>1 else (float('inf'), None)
        u_future = float(u_future)
        i_future, i_next = inherited[1] if len(inherited)>1 else (float('inf'), None)
        i_future = float(i_future)

        ms_lst = []
        ms = int(obj.strip().split(",")[2])
        ms_lst.append(ms)

        if u_future <= ms:
            u_point = u_next
            uninherited.pop(0)
        i_point = i_point if i_time < ms else None
        if i_future <= ms:
            i_point = i_next
            inherited.pop(0)

        #determine ms based on type
        t, slides = find_type(obj)
        complete_time = -1
        if t.startswith('slider'):
            s_len = find_len(obj)
            beatlen = tpoint_beatlen(u_point)
            sv = tpoint_sv(i_point) if i_point else 1
            
            complete_time = s_len / (slider_mult * 100 * sv) * beatlen 
            slides_ms = []
            for i in range(1, slides+1):
                slides_ms.append(ms+i*complete_time)
            ms_lst.extend(slides_ms)
            
            #check for timing points at end of slider
            i_future, i_next = inherited[1] if len(inherited)>1 else (float('inf'), None)
            i_future = float(i_future)
            if i_future <= ms+complete_time:
                i_point=i_next
                inherited.pop(0)
            u_future, u_next = uninherited[1] if len(uninherited)>1 else (float('inf'), None)
            u_future = float(u_future)
            if u_future <= ms+complete_time:
                u_point=u_next
                uninherited.pop(0)

        if ms==1703: #TODO: wrong complete time calculation for we are dreamers slider at 1703 timestamp
            print(f"complete_time components are s_len:{s_len}, slider_mult:{slider_mult}, sv:{sv}, beatlen:{beatlen}")
            print(f"complete_time is {complete_time}")
            print(f"number of slides is {slides}")

        if t.startswith('spinner'):
            ms_lst = []
            end_time = find_end_time(obj)
            ms_lst.append(end_time)

        return ms_lst, complete_time

    def hitobject_seq_to_ms(self, seq, slider_mult, uninherited, inherited):
        ms_seq = []
        delta_forward = []
        delta_backward = []
        nxt = None
        prev = None

        for i, obj in enumerate(seq):
            if i < len(seq)-1:
                nxt = seq[i+1]
            else:
                nxt = None
            ms_lst, complete_time = self.hitobject_to_ms(obj, slider_mult, uninherited, inherited)
            ms_seq.extend(ms_lst)
            forward, backward = self.timing_deltas(prev, obj, nxt, complete_time)
            delta_forward.extend(forward)
            delta_backward.extend(backward)
            prev = obj
        return ms_seq, delta_forward, delta_backward



