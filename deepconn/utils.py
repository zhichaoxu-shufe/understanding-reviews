import torch
import torch.nn.functional as F 

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def masked_softmax(input_scores, input_masks):
	"""
	Args:
		inputs_scores: [bz, seq_len_a, seq_len_b]
		input_masks: [bz, 1, seq_len_b]

		or 

		input_scores: [bz, seq_len]
		input_masks: [bz, seq_len]

	Returns:
		output_weights: [bz, seq_len]
	"""
	assert input_scores.dim() == input_masks.dim()
	return F.softmax(torch.masked_fill(input_scores, ~input_masks, -1e8), dim=-1)

def masked_colwise_mean(inputs, input_masks):
	"""
	Args:
		input_scores: [bz, row, col]
		input_masks: [bz, 1, col]

	Returns:
		outputs: [bz, row, 1]
	"""
	"""
	assert input_scores.dim() == input_masks.dim()
	input_scores = input_scores.sum(dim=-1, keepdim=True) #[bz, row, 1]

	int_masks = input_masks.float() 
	input_lenghts = int_masks.sum(dim=-1, keepdim=True) #[bz, 1, 1]

	return input_scores / (input_lenghts + 1.)
	"""
	assert input_masks.dim() == 3
   
	float_masks = input_masks.float() #[bz, 1, seq_len]
	input_lengths = float_masks.sum(dim=2, keepdim=True) + 1e-8  #[bz, 1, 1]
	sum_inputs = (inputs * float_masks).sum(dim=2, keepdim=True) #[bz, hdim, 1]
	return sum_inputs / input_lengths
	



def masked_tensor(inputs, masks):
	"""
	Args:
		inputs: [*, hidden_dim]
		masks: [*]
		* should have same shape
	Ouputs:
		masked_inputs: [*, hidden_dim]
	"""
	assert inputs.shape[:-1] == masks.shape
	masks = masks.unsqueeze(-1)
	masked_inputs = torch.masked_fill(inputs, ~masks, 0.)
	return masked_inputs

def attention_weighted_sum(input_weights, inputs):
	"""
	Args:
		input_weights: [bz, seq_len] or [bz, seq_len, 1]
		inputs: [bz, seq_len, hidden_dim]
	
	Returns:
		outputs: [bz, hidden_dim]
	"""
	if input_weights.dim() == 2:
		input_weights = input_weights.unsqueeze(-1)
	
	outputs = torch.sum(input_weights * inputs, dim=1)
	return outputs

def get_mask(tensor, padding_idx=0):
	"""
	Get a mask to `tensor`.
	Args:
		tensor: LongTensor with shape of [bz, seq_len]

	Returns:
		mask: BoolTensor with shape of [bz, seq_len]
	"""
	mask = torch.ones(size=list(tensor.size()), dtype=torch.bool)
	mask[tensor == padding_idx] = False 

	return mask 

def get_seq_lengths_from_mask(mask_tensor):
	"""
	NOTE: Not generalize, just deal with a special condition where
	mask_tensor: BoolTensor with shape of [bz, review_num, sent_num, word_num]
	length_tensor: LongTensor with shape of [bz, review_num, sent_num]
	"""
	int_tensor = mask_tensor.int()
	length_tensor = int_tensor.sum(dim=-1)
   
	return length_tensor

def get_rev_mask(inputs):
		"""
		If rv_len are all 0, then corresponding position in rv_num should be 0
		Args:
			inputs: [bz, rv_num, rv_len]
		"""
		bz, rv_num, _ = list(inputs.size())

		masks = torch.ones(size=(bz, rv_num)).int()
		inputs = inputs.sum(dim=-1) #[bz, rv_num]
		masks[inputs==0] = 0 

		return masks.bool()


if __name__ == "__main__":
	"""
	x = torch.BoolTensor([[[1,1,0,0],[1,0,0,0], [1,1,1,0]],
							[[1,1,1,1], [1,0,0,0], [1,1,0,0]]])
	y = get_seq_lengths_from_mask(x)
	print("bool tensor")
	print(x)
	print("length tensor")
	print(y)

	x = torch.LongTensor([[[7,8,2,0],[1,4,5,0], [2,3,4,5]],
							[[3,3,2,1], [1,0,0,0], [1,1,0,0]]])
	y = get_mask(x)

	print("tensor")
	print(x, x.shape)
	print("corresponding mask")
	print(y)

	x  = torch.randn(2, 5, 10)
	mask = torch.tensor([
		[1,1,1,0,0],
		[1,0,0,0,0]
	]).bool()
	print("masked inputs", x, masked_inputs(x, mask))
	"""
	x = torch.tensor([[4,5,6], [2,3,3], [7,2,0], [0,0,0]]).view(1,4,3)
	y = torch.tensor([[4,5,6], [1,0,0], [0,0,0], [0,0,0]]).view(1,4,3)
	inp = torch.cat([x,y],dim=0)
	print(inp)
	print(get_rev_mask(inp), get_rev_mask(inp).shape)
