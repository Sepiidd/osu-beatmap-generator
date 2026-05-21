from hitobject-utils import find_type

class obj_converter():
	def __init__(self, special_start=["<CIRCLE>", "<SLIDER_HEAD_BEZIER>", "<SLIDER_HEAD_LINEAR>", "<SLIDER_HEAD_PERFECT>"], tokenizer):
		self.special_start = special_start
        self.tokenizer = tokenizer

	def tok_to_hitobject(self, obj_arr):
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
		seq_obj = []
	 	idx = 0
		curr = []
		while idx < len(seq):
			if len(curr) == 0: #determine current object type (circle, slider-b/l/p)
				curr.append(seq[idx])
			elif seq[idx] not in self.special_start:
				curr.append(seq[idx])
			else:
				obj = token_to_hitobject(curr)
				seq_obj.append(obj)
				curr=[seq[idx]]
		return seq_obj

	def hitobject_to_tok(self, obj):
        tokens = []
        obj_type, slides = find_type(obj)
        separated = obj_type.strip().split(",")
        if obj_type == "circle":
            name = "<CIRCLE>"
            x = f"X_{separated[0]}"
            y = f"Y_{separated[1]}"
            tokens.extend([name, x, y])
        elif obj_type.startswith("slider"):
            name = ""
            match obj_type:
                case "slider-b":
                    name = "<SLIDER_HEAD_BEZIER>"
                case "slider-l":
                    name = "<SLIDER_HEAD_LINEAR>"
                case "slider-p":
                    name = "<SLIDER_HEAD_PERFECT>"
            x = f"X_{separated[0]}"
            y = f"Y_{separated[1]}"
            tokens.extend([name, x, y])
            anchor_s = seperated[5].split("|")[1:]
            anchor_arr = []
            for anchor in anchor_s:
                anchor_name = "<ANCHOR>"
                pos = anchor.split(":")
                anchor_x = pos[0]
                anchor_y = pos[1]
                anchor_arr.extend([anchor_name, anchor_x, anchor_y])
            tokens.extend(anchor_arr)
        return tokens
	
    def hitobject_seq_to_tok(self, seq): #TODO: change to take \n seperated string or something
		tokens = []
		for obj in seq:
			tokens.extend(hitobject_to_tok(obj))
		return tokens
