from my_tokenizer import train_tokenizer,Tokenizer, tokenizer
from imports import *
from MultiHeadAttention import MHA
from POS import POS_emb
from Encoder import _Encoder

tokenizer_path      = "./My_GPT/self_trained_tokenizer/"
dataset             = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
tokenizer_train     = train_tokenizer()

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        return dataset[i : i + 1000]["text"]



tokenizer_train.train(get_training_corpus())


test_sent = ["BET Chairman and CEO Debra Lee Is Stepping Down",
"Rachel Dolezal Faces Felony Charges For Welfare Fraud",
"Bishop Michael Curry Joins Christian March To White House To 'Reclaim Jesus'",
"Kanye West Spent $85,000 On Photo Of Whitney Houston's Bathroom For Album Cover",
"Morgan Freeman Dropped From Marketing Campaigns After Harassment Accusations"]

'''
Encode sentences to their corresponding token ids from vocab
'''
token_idx = tokenizer.encode_batch(test_sent)

d_mod = 512
d_ff = 2048
pos_emb = POS_emb(tokenizer,token_idx,d_mod).generate()

encoder_obj = _Encoder(pos_emb,d_mod,d_ff)

encoder_obj.generate()

dummy = 0