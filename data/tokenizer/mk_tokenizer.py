from tokenizers import Tokenizer 
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer 
from tokenizers.pre_tokenizers import Whitespace 

tokenizer = Tokenizer(BPE(unk_token="<unk>",))
trainer = BpeTrainer(special_tokens=["<unk>", 
    "<pad>", "<mask>", "<s>", "</s>"])

tokenizer.pre_tokenizer = Whitespace()
files = ["../raw_data/tokenizer/wmt2014_train.en"]
tokenizer.train(files, trainer)

tokenizer.save("en_tokenizer.json")
