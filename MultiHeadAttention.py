from imports import *



class MHA():

    def __init__(self,d_model : int, heads : int, seq_len : int) -> None:
        
        self.d_k            = d_model
        self.h              = heads
        self.seq_len        = seq_len
        self.softmax        = torch.nn.Softmax(dim=2)
        
        assert d_model % heads == 0
        
        self.multi_head = d_model // heads
        self.W = torch.nn.Linear(d_model,d_model)

    def att_score(query, key, value,dropout = torch.nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        # if dropout is not None:
        #     attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores


    def calculate(self,emb_mat):

        key         = self.W(emb_mat)
        query       = self.W(emb_mat)
        value       = self.W(emb_mat)

        # Break the original full matrix into heads
        query       = query.view(query.shape[0], query.shape[1], self.h, self.multi_head).transpose(1, 2)
        key         = key.view(key.shape[0], key.shape[1], self.h, self.multi_head).transpose(1, 2)
        value       = value.view(value.shape[0], value.shape[1], self.h, self.multi_head).transpose(1, 2)

        att_score_mat, self.attention_scores = MHA.att_score(query, key, value)

        att_score_mat = att_score_mat.transpose(1,2).contiguous().view(att_score_mat.shape[0],-1,self.multi_head * self.h)

        return self.W(att_score_mat),self.attention_scores
