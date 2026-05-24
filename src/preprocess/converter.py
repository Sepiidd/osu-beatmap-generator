from preprocess.hitobject_utils import find_type

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
