from imports import *
from MultiHeadAttention import MHA
from Add_Norm import add_norm


class _Encoder():

    def __init__(self,final_pos_emb,d_mod,d_ff) -> None:
        
        self.emb_mat = final_pos_emb
        self.d_mod = d_mod
        self.d_ff = d_ff

    def multi_head_att_block(self):

        multi_head_attention = MHA(512,4,self.emb_mat[0].shape[0])

        mha_att_scores, attention_score_mat = multi_head_attention.calculate(torch.stack(self.emb_mat,0))
    
        return mha_att_scores
    
    def add_and_norm(self,last_layer,curr_layer):

        return add_norm(last_layer,curr_layer,0.2).res_net()
    
    def fforward(self,x):

        l1 = torch.nn.Linear(self.d_mod,self.d_ff)
        l2 = torch.nn.Linear(self.d_ff, self.d_mod)
        dropout = torch.nn.Dropout(0.2)
        return l2(dropout(torch.relu(l1(x))))


    def generate(self):

        mha_out = self.multi_head_att_block()

        add_norm_1 = self.add_and_norm(self.emb_mat[0],mha_out)

        ffw_out = self.fforward(add_norm_1)

        add_norm_2 = self.add_and_norm(add_norm_1,ffw_out)