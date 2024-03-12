from imports import *
from MultiHeadAttention import MHA
from Add_Norm import add_norm


class _Decoder_block(torch.nn.Module):

    def __init__(self,seq_len,d_mod,d_ff,heads) -> None:
        super(_Decoder_block, self).__init__()
        
        self.d_mod          = d_mod
        self.d_ff           = d_ff
        self.heads          = heads
        self.multi_head_attention = MHA(self.d_mod,self.heads,seq_len)
        self.l1 = torch.nn.Linear(self.d_mod,self.d_ff)
        self.l2 = torch.nn.Linear(self.d_ff, self.d_mod)
        self.dropout = torch.nn.Dropout(0.2)

    
    def add_and_norm(self,last_layer,curr_layer):

        return add_norm(last_layer,curr_layer,0.2).res_net()
    
    def fforward(self,x):

        return self.l2(self.dropout(torch.relu(self.l1(x))))

    def cross_attention(self,add_norm_1,enc_output,enc_pad_mask,dec_pad_mask):
        mha_att_scores, attention_score_mat = self.multi_head_attention.calculate(add_norm_1,enc_output = enc_output,cross_att = [enc_pad_mask,dec_pad_mask])
        return mha_att_scores
        

    def generate(self,emb_mat,enc_output,dec_pad_mask,enc_pad_mask):

        mha_out,_ = self.multi_head_attention.calculate(emb_mat,dec_pad_mask=dec_pad_mask) # masked self attention

        add_norm_1 = self.add_and_norm(emb_mat,mha_out)

        cross_out = self.cross_attention(add_norm_1,enc_output,enc_pad_mask,dec_pad_mask)
        
        add_norm_2 = self.add_and_norm(add_norm_1,cross_out)

        ffw_out = self.fforward(add_norm_2)

        add_norm_3 = self.add_and_norm(add_norm_2,ffw_out)

        return add_norm_3

class Whole_decoder(torch.nn.Module):
    
    def __init__(self,no_of_layers,d_mod,d_ff,heads,seq_len) -> None:
        super(Whole_decoder, self).__init__()
        self.layers      = no_of_layers
        self.d_mod       = d_mod
        self.d_ff        = d_ff
        self.heads       = heads
        self.decoder     = _Decoder_block(seq_len,self.d_mod,self.d_ff,self.heads)

    def forward(self,encoder_out,enc_output,dec_pad_mask,enc_pad_mask):
        
        for _ in range(self.layers):
            encoder_out    = self.decoder.generate(encoder_out,enc_output,dec_pad_mask,enc_pad_mask)

        return encoder_out