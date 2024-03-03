from imports import *
from MultiHeadAttention import MHA
from Add_Norm import add_norm


class _Decoder_block():

    def __init__(self,final_pos_emb,d_mod,d_ff,heads,enc_output) -> None:
        
        self.emb_mat        = final_pos_emb
        self.d_mod          = d_mod
        self.d_ff           = d_ff
        self.heads          = heads
        self.enc_output     = enc_output

    def multi_head_att_block(self,decoder):

        multi_head_attention = MHA(self.d_mod,self.heads,self.emb_mat[0].shape[0])

        mha_att_scores, attention_score_mat = multi_head_attention.calculate(self.emb_mat,decoder=decoder)
    
        return mha_att_scores
    
    def add_and_norm(self,last_layer,curr_layer):

        return add_norm(last_layer,curr_layer,0.2).res_net()
    
    def fforward(self,x):

        l1 = torch.nn.Linear(self.d_mod,self.d_ff)
        l2 = torch.nn.Linear(self.d_ff, self.d_mod)
        dropout = torch.nn.Dropout(0.2)
        return l2(dropout(torch.relu(l1(x))))

    def cross_attention(self,add_norm_1):
        multi_head_attention = MHA(self.d_mod,self.heads,add_norm_1[0].shape[0])
        mha_att_scores, attention_score_mat = multi_head_attention.calculate(add_norm_1,enc_output = self.enc_output)
        return mha_att_scores
        

    def generate(self):

        mha_out = self.multi_head_att_block(decoder=True) # masked self attention

        add_norm_1 = self.add_and_norm(self.emb_mat,mha_out)

        cross_out = self.cross_attention(add_norm_1)
        
        add_norm_2 = self.add_and_norm(add_norm_1,cross_out)

        ffw_out = self.fforward(add_norm_2)

        add_norm_3 = self.add_and_norm(add_norm_2,ffw_out)

        return add_norm_3

class Whole_decoder():
    
    def __init__(self,no_of_layers,final_pos_emb,d_mod,d_ff,heads,enc_output) -> None:
        self.layers      = no_of_layers
        self.encoder_out = torch.stack(final_pos_emb,0)
        self.d_mod       = d_mod
        self.d_ff        = d_ff
        self.heads       = heads
        self.enc_output  = enc_output

    def forward(self):
        
        for _ in range(self.layers):
            block               = _Decoder_block(self.encoder_out,self.d_mod,self.d_ff,self.heads,self.enc_output)
            self.encoder_out    = block.generate()

        return self.encoder_out