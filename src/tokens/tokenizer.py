import json
from pathlib import Path
BASE_DIR=Path(__file__).parent

class obg_tokenizer:
	def __init__(
	self, 
	load_tokens=False, 
	special_start=["<CIRCLE>", "<SLIDER_HEAD_BEZIER>", "<SLIDER_HEAD_LINEAR>", "<SLIDER_HEAD_PERFECT>"], 
	special_types=["<ANCHOR>", "<SLIDER_TAIL>, <SLIDER_REPEAT>"], 
	x_min=-500,     #ingame grid min -180
	x_max=1000,      #ingame grid max 691
	y_min=-500,      #ingame grid min -82
	y_max=1000):     #ingame grid max 407
		if load_tokens:
			self.load_tokens()
			return 

		self.special = special_start
		self.special.extend(special_types)
		self.special.extend(["<BOS>", "<EOS>", "<PAD>"])

		self.special_start = special_start #tokens which indicate a new hitobject/type
		self.special_types = special_types #other special tokens

		#include max value in positions
		self.x_from = x_min
		self.x_positions = x_max + 1
		self.y_from = y_min
		self.y_positions = y_max + 1

		self.itot={}
		self.ttoi={}

		index = 0
		for s in self.special:
			self.itot[index] = s
			self.ttoi[s] = index
			index+=1
		for i in range(self.x_from, self.x_positions):
			self.itot[index] = f"X_{i}"
			self.ttoi[f"X_{i}"] = index
			index+=1
		for i in range(self.y_from, self.y_positions):
			self.itot[index] = f"Y_{i}"
			self.ttoi[f"Y_{i}"] = index
			index+=1

		self.num_t = index+1

	#----------save+load tokens json----------
	def save_tokens(self):
		with open(BASE_DIR / "itot.json", "w") as f:
			json.dump(self.itot, f)
		with open(BASE_DIR / "ttoi.json", "w") as f:
			json.dump(self.ttoi, f)
		with open(BASE_DIR / "meta.json", "w") as f:
			json.dump({
				"special_start": self.special_start, 
				"special_types": self.special_types, 
				"special": self.special, 
				"x_from": self.x_from,
				"x_positions": self.x_positions, 
				"y_from": self.y_from,
				"y_positions": self.y_positions, 
				"num_t": self.num_t}, 
				f
			)

	def load_tokens(self):
		with open(BASE_DIR / "itot.json", "r") as f:
			self.itot = json.load(f)
		with open(BASE_DIR / "ttoi.json", "r") as f:
			self.ttoi = json.load(f)
		with open(BASE_DIR / "meta.json", "r") as f:
			meta = json.load(f)
			self.special = meta['special']
			self.special_start = meta['special_start']
			self.special_types = meta['special_types']
			self.x_from = meta['x_from']
			self.x_positions = meta['x_positions']
			self.y_from = meta['y_from']
			self.y_positions = meta['y_positions']
			self.num_t = meta['num_t']

	#----------encode+decode----------
	def encode(self, obj):
		return self.ttoi[obj]
	
	def encode_seq(self, seq):
		seq_ids = []
		for obj in seq:
			seq_ids.append(self.encode(obj))
		return seq_ids

	def decode(self, idx):
		return self.itot[idx]

	def decode_seq(self, seq):
		seq_tok = []
		for idx in seq:
			seq_tok.append(self.decode(idx))
		return seq_tok

if __name__ == "__main__":
	tokenizer = obg_tokenizer()
	print("num_t is:", tokenizer.num_t)
	print("special is:", tokenizer.special)
	tokenizer.save_tokens()

