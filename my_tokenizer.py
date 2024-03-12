from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


'''
we’ll create a Tokenizer with a WordPiece model:
'''
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
'''
Using Bert normalizer and tokenizer(WHitespace, punctuation, )
'''
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

'''
innitalise a word peice trainer
'''
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)



class train_tokenizer():
    # def __init__(self,raw_corpus,test_text):
    #     self.raw_corpus = raw_corpus
    #     self.test_text = test_text

    def train(self,raw_corpus):
        tokenizer.train_from_iterator(raw_corpus, trainer=trainer)
        cls_token_id = tokenizer.token_to_id("[CLS]")
        sep_token_id = tokenizer.token_to_id("[SEP]")
        pad_token_id = tokenizer.token_to_id("[PAD]")
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token="[PAD]")
        
        tokenizer.post_processor = processors.TemplateProcessing(
                                    single=f"[CLS]:0 $A:0 [SEP]:0",
                                    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
                                    special_tokens=[
                                        ("[CLS]", cls_token_id),
                                        ("[SEP]", sep_token_id),
                                    ],
                                )
    
    def save_self_trained(self,path):
        tokenizer.save(path+"config.json")