from imports import *
from MultiHeadAttention import MHA
from Add_Norm import add_norm


class _Encoder_block(torch.nn.Module):

    def __init__(self,d_mod,d_ff,heads,seq_len) -> None:
        super(_Encoder_block, self).__init__()
        
        self.d_mod          = d_mod
        self.d_ff           = d_ff
        self.heads          = heads
        self.multi_head_attention = MHA(self.d_mod,self.heads,seq_len)
        self.l1 = torch.nn.Linear(self.d_mod,self.d_ff)
        self.l2 = torch.nn.Linear(self.d_ff, self.d_mod)
        self.dropout = torch.nn.Dropout(0.2)

    def multi_head_att_block(self,emb_mat,pad_mask):

        mha_att_scores, attention_score_mat = self.multi_head_attention.calculate(emb_mat,pad_mask=pad_mask)
    
        return mha_att_scores
    
    def add_and_norm(self,last_layer,curr_layer):

        return add_norm(last_layer,curr_layer,0.2).res_net()
    
    def fforward(self,x):
        
        return self.l2(self.dropout(torch.relu(self.l1(x))))


    def generate(self,emb_mat,pad_mask):

        mha_out = self.multi_head_att_block(emb_mat,pad_mask)

        add_norm_1 = self.add_and_norm(emb_mat,mha_out)

        ffw_out = self.fforward(add_norm_1)

        add_norm_2 = self.add_and_norm(add_norm_1,ffw_out)

        return add_norm_2

class Whole_encoder(torch.nn.Module):
    
    def __init__(self,no_of_layers,d_mod,d_ff,heads,seq_len) -> None:
        super(Whole_encoder, self).__init__()

        self.layers      = no_of_layers
        self.d_mod       = d_mod
        self.d_ff        = d_ff
        self.heads       = heads
        self.block       = _Encoder_block(self.d_mod,self.d_ff,self.heads,seq_len)


    def forward(self,encoder_out,pad_mask):
        
        for _ in range(self.layers):
            encoder_out    = self.block.generate(encoder_out,pad_mask)

        return encoder_out