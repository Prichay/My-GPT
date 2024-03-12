from my_tokenizer import train_tokenizer,Tokenizer, tokenizer
from imports import *
from MultiHeadAttention import MHA
from POS import POS_emb
from Encoder import Whole_encoder
from Decoder import Whole_decoder
from projection import projectionLayer



def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        return english+hinglish[i : i + 1000]
    

def add_srt_end_token(decoder_batch):

    return ["<SOS> "+i+" <EOS>" for i in decoder_batch]

    
dataset               = load_dataset("findnitai/english-to-hinglish",split='train')
tokenizer_path        = "./My_GPT/self_trained_tokenizer/"
tokenizer_train       = train_tokenizer()
dataset_dict          = dataset.to_dict()
english               = [item['en'] for item in dataset_dict['translation']]
hinglish              = [item['hi_ng'] for item in dataset_dict['translation']]
tokenizer_train       .train(get_training_corpus())





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




class Transformer(torch.nn.Module):

    def __init__(self,tokenizer, d_mod, d_ff, no_of_layers, heads) -> None:
        super(Transformer, self).__init__()
        self.tokenizer            = tokenizer
        self.d_mod                = d_mod
        self.d_ff                 = d_ff
        self.no_of_layers         = no_of_layers
        self.heads                = heads
        self.POS_obj              = POS_emb(self.tokenizer,self.d_mod)
        self.encoder_obj          = Whole_encoder(self.no_of_layers,self.d_mod,self.d_ff,self.heads)
        self.decoder_obj          = Whole_decoder(self.no_of_layers,self.d_mod,self.d_ff,self.heads)
        self.projection_layer     = projectionLayer(self.d_mod,tokenizer.get_vocab_size())



    def forward(self,token_idx_encode,token_idx_decode ):


        pos_emb_encoder     = self.POS_obj.generate(token_idx_encode)
        pos_emb_decoder     = self.POS_obj.generate(token_idx_decode)

        att_mask_tensor     = torch.stack([torch.Tensor(i.attention_mask) for i in token_idx_encode])
        enc_pad_mask        = [(i == 0).type(torch.int16).unsqueeze(0) for i in att_mask_tensor]

        att_mask_tensor_decoder     = [torch.Tensor(i.attention_mask) for i in token_idx_decode]
        dec_pad_mask        = [(i == 0).type(torch.int16).unsqueeze(0) for i in att_mask_tensor_decoder]

        enc_output          = self.encoder_obj.forward(torch.stack(pos_emb_encoder,0),enc_pad_mask)
        decoder_out         = self.decoder_obj.forward(torch.stack(pos_emb_decoder,0),enc_output,dec_pad_mask,enc_pad_mask)
        prob_dist           = self.projection_layer.project(decoder_out)

        return prob_dist,token_idx_decode




d_mod           = 512
d_ff            = 2048
no_of_layers    = 4
heads           = 4
prev_batch      = 0
max_seq_len     = 100
jump            = 50
criterion       = torch.nn.CrossEntropyLoss()
model           = Transformer(tokenizer,d_mod,d_ff,no_of_layers,heads)

bad_index       = [*set([index for index,i in enumerate(english) if len(i) > 200] + [index for index,i in enumerate(hinglish) if len(i) > 200])]
english         = [i for index,i in enumerate(english) if index not in bad_index]
hinglish        = [i for index,i in enumerate(hinglish) if index not in bad_index]



for batch in range(50,5000,jump):
    test_sent       = english[prev_batch : batch]
    decoder_text    = hinglish[prev_batch : batch]
    # add SOS and EOS to every sentance
    decoder_text    = add_srt_end_token(decoder_text)
    print(f" Batch from {prev_batch} to {batch}")
    prev_batch      = batch



    encoding                          = tokenizer.encode_batch(test_sent+decoder_text)
    token_idx_encode,token_idx_decode = (encoding[:jump],encoding[jump:])
    prob_dist,token_idx_decode        = model.forward(token_idx_encode,token_idx_decode)
    optimizer                         = torch.optim.Adam(model.parameters(), lr=0.0001)

    #################### LOSS CALC
    one_hot            = torch.stack([torch.nn.functional.one_hot(torch.tensor(i.ids), num_classes=tokenizer.get_vocab_size()) for i in token_idx_decode])
    target_flat        = one_hot.view(-1, tokenizer.get_vocab_size())
    prob_dist_flat     = prob_dist.view(-1, tokenizer.get_vocab_size())
    loss               = criterion(prob_dist_flat, torch.argmax(target_flat, dim=1))

    print(f"\nLoss : {loss}\n")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# print([k for k,v in tokenizer.get_vocab().items() if v == prob_dist[0][2].tolist().index(max(prob_dist[0][2].tolist()))])

# for words,text in zip(prob_dist[0],decoder_text[0].split()):
#     for k,v in tokenizer.get_vocab().items():
#         if v == words.tolist().index(max(words.tolist())):
#             print(f"word is : {text}\npredicted word is {k}\n\n")