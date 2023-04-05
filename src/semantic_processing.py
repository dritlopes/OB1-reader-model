from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
import torch

sequence = "AI can do great"
# initialize model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# in case you want to reproduce predictions
set_seed(42)
# pre-process text
encoded_input = tokenizer(sequence, return_tensors='pt')
# get logits
print(sequence + '...')
# output contains at minimum the prediction scores of the language modelling head,
# i.e. scores for each vocab token given by a feed-forward neural network
output = model(**encoded_input)
# logits are prediction scores of language modelling head; of shape (batch_size, sequence_length, config.vocab_size)
logits = output.logits[:, -1, :]
pred_word = tokenizer.decode([torch.argmax(logits).item()])
print('top 1 prediction: ', pred_word)
topk_words = [tokenizer.decode(id.item()) for id in torch.topk(logits, k=10)[1][0]]
print('top 10 predictions: ', topk_words)