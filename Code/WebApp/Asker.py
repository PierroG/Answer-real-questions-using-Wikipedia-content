import torch
import transformers
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer


class ask:
	def __init__(self):
		self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
		self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
		self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
		print(self.device)
		#move model to device
		self.model = self.model.to(self.device)

	def generateAnswer(self, question, answer_text):
		print("I'm looking for an aswer, wait please ...")
		# == Tokenize == Apply the tokenizer to the input text, treating them as a text-pair. (CPU)
		print("-Tokenization")
		input_ids = self.tokenizer.encode(question, answer_text)
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
		# == Set Segment IDs == Search the input_ids for the first instance of the `[SEP]` token.
		sep_index = input_ids.index(self.tokenizer.sep_token_id)
		# The number of segment A tokens includes the [SEP] token istelf.
		num_seg_a = sep_index + 1
		# The remainder are segment B.
		num_seg_b = len(input_ids) - num_seg_a
		# Construct the list of 0s and 1s.
		segment_ids = [0]*num_seg_a + [1]*num_seg_b
		# There should be a segment_id for every input token.
		assert len(segment_ids) == len(input_ids)
		# == Run Model == Run our example through the model. (GPU)
		print("-Forward pass on the model")
		#move tensor to device
		input_ids_tensor = torch.tensor([input_ids]).to(self.device)
		segment_ids_tensor = torch.tensor([segment_ids]).to(self.device)
	
		start_scores, end_scores = self.model(input_ids_tensor, # The tokens representing our input text.
								 token_type_ids=segment_ids_tensor) # The segment IDs to differentiate question from 
		# Find the tokens with the highest `start` and `end` scores.
		answer_start = torch.argmax(start_scores)
		answer_end = torch.argmax(end_scores)
		# get score
		start_score = float(start_scores[0,answer_start])
		end_score = float(end_scores[0,answer_end])
	
	
		# == Print Answer without ## ==
		# Start with the first token.
		answer = tokens[answer_start]

		# Select the remaining answer tokens and join them with whitespace.
		for i in range(answer_start + 1, answer_end + 1):
	
			# If it's a subword token, then recombine it with the previous token.
			if tokens[i][0:2] == '##':
				answer += tokens[i][2:]
	
			# Otherwise, add a space then the token.
			else:
				answer += ' ' + tokens[i]

		return answer, start_score+end_score


if __name__=='__main__':
	asker = ask()
	a,s = asker.generateAnswer('who i am ?', 'i am me')
	print(a)

