from my_tokenizer import train_tokenizer,Tokenizer, tokenizer
from imports import *
from MultiHeadAttention import MHA
from POS import POS_emb
from Encoder import Whole_encoder
from Decoder import Whole_decoder
from projection import projectionLayer

tokenizer_path      = "./My_GPT/self_trained_tokenizer/"
dataset             = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
tokenizer_train     = train_tokenizer()

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        return dataset[i : i + 1000]["text"]



tokenizer_train.train(get_training_corpus())

decoder_text = [
    "BET Chairman aur CEO Debra Lee resign kar rahi hain.",
    "Rachel Dolezal ko Welfare Fraud ke liye Felony charges ka samna karna pad raha hai.",
    "Bishop Michael Curry White House ki or 'Reclaim Jesus' Christian March mein shaamil ho rahe hain.",
    "Kanye West ne Whitney Houston ki bathroom ki photo ke liye album cover ke liye $85,000 kharch kiye hain.",
    "Morgan Freeman ko harassment ke aarop ke baad marketing campaigns se hata diya gaya hai."
]

test_sent = ["BET Chairman and CEO Debra Lee Is Stepping Down",
"Rachel Dolezal Faces Felony Charges For Welfare Fraud",
"Bishop Michael Curry Joins Christian March To White House To 'Reclaim Jesus'",
"Kanye West Spent $85,000 On Photo Of Whitney Houston's Bathroom For Album Cover",
"Morgan Freeman Dropped From Marketing Campaigns After Harassment Accusations"]

'''
Encode sentences to their corresponding token ids from vocab
'''
token_idx_encode = tokenizer.encode_batch(test_sent)

token_idx_decode = tokenizer.encode_batch(decoder_text)


d_mod               = 512
d_ff                = 2048
pos_emb_encoder     = POS_emb(tokenizer,token_idx_encode,d_mod).generate()
pos_emb_decoder     = POS_emb(tokenizer,token_idx_decode,d_mod).generate()
no_of_layers        = 4
heads               = 4
encoder_obj         = Whole_encoder(no_of_layers,pos_emb_encoder,d_mod,d_ff,heads)
enc_output          = encoder_obj.forward()
decoder_obj         = Whole_decoder(no_of_layers,pos_emb_decoder,d_mod,d_ff,heads,enc_output)
decoder_out         = decoder_obj.forward()
projection_layer    = projectionLayer(d_mod,tokenizer.get_vocab_size())
prob_dist           = projection_layer.project(decoder_out)


# print([k for k,v in tokenizer.get_vocab().items() if v == prob_dist[0][2].tolist().index(max(prob_dist[0][2].tolist()))])

for words,text in zip(prob_dist[0],decoder_text[0].split()):
    for k,v in tokenizer.get_vocab().items():
        if v == words.tolist().index(max(words.tolist())):
            print(f"word is : {text}\npredicted word is {k}\n\n")