from transformers import PreTrainedTokenizerFast as Tokenizer

tokenizer = Tokenizer(tokenizer_file="en_tokenizer.json")

text = "the quick brown fox jumps over the lazy dog?"

print(tokenizer.decode(tokenizer.encode(text)))


