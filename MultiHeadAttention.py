from imports import *

def create_look_ahead_mask(size):
    # Create a lower triangular matrix with ones above the main diagonal
    look_ahead_mask = torch.triu(torch.ones(size, size), diagonal=1)
    return look_ahead_mask == 0


class MHA(torch.nn.Module):

    def __init__(self,d_model : int, heads : int, seq_len : int) -> None:
        super(MHA, self).__init__()
        
        self.d_k            = d_model
        self.h              = heads
        self.seq_len        = seq_len
        self.softmax        = torch.nn.Softmax(dim=2)
        
        assert d_model % heads == 0
        
        self.multi_head = d_model // heads
        self.W = torch.nn.Linear(d_model,d_model)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def att_score(self,query, key, value,pad_mask = False,dec_pad_mask = False,cross_att = False):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        if pad_mask:
            pad_masked_att_score = (query @ key.transpose(-2, -1)).masked_fill(torch.stack(pad_mask).unsqueeze(-1).expand((query @ key.transpose(-2, -1)).shape) ==1,value = -1e9)
            attention_scores = pad_masked_att_score / math.sqrt(d_k)
        elif dec_pad_mask:
            look_ahead = torch.broadcast_to(self._generate_square_subsequent_mask(self.seq_len),(query @ key.transpose(-2, -1)).shape)
            dec_pad_mask = torch.stack(dec_pad_mask).unsqueeze(3).expand_as(look_ahead)
            combined_mask = look_ahead.logical_or(dec_pad_mask).int()
            combined_mask = combined_mask.masked_fill(combined_mask == 0, float(1.)).masked_fill(combined_mask == 1, float(0.0))
            attention_scores = ((query @ key.transpose(-2, -1)).masked_fill(combined_mask == 0,value = -1e9)) / math.sqrt(d_k)
        elif cross_att:
            look_ahead_query = torch.broadcast_to(self._generate_square_subsequent_mask(self.seq_len),(query @ key.transpose(-2, -1)).shape)
            enc_mask = torch.stack(cross_att[0])
            enc_pad_mask = enc_mask.unsqueeze(3).expand_as(look_ahead_query)
            combined_mask = look_ahead_query.logical_or(enc_pad_mask).int()
            combined_mask = combined_mask.masked_fill(combined_mask == 0, float(1.)).masked_fill(combined_mask == 1, float(0.0))
            attention_scores = ((query @ key.transpose(-2, -1)).masked_fill(combined_mask == 0,value = -1e9)) / math.sqrt(d_k)

            
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        # if dropout is not None:
        #     attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores


    def calculate(self,emb_mat,enc_output = False,pad_mask = False,dec_pad_mask=False, cross_att = False):

        if type(enc_output) == torch.Tensor:
            key     = self.W(enc_output)
            value   = self.W(enc_output)
        else:
            key         = self.W(emb_mat)
            value       = self.W(emb_mat)

        query       = self.W(emb_mat)

        # Break the original full matrix into heads
        query       = query.view(query.shape[0], query.shape[1], self.h, self.multi_head).transpose(1, 2)
        key         = key.view(key.shape[0], key.shape[1], self.h, self.multi_head).transpose(1, 2)
        value       = value.view(value.shape[0], value.shape[1], self.h, self.multi_head).transpose(1, 2)
        
        if dec_pad_mask:
            att_score_mat, self.attention_scores = self.att_score(query, key, value,dec_pad_mask=dec_pad_mask)
        elif pad_mask:
            att_score_mat, self.attention_scores = self.att_score(query, key, value,pad_mask = pad_mask)
        elif cross_att:
            att_score_mat, self.attention_scores = self.att_score(query, key, value,cross_att=cross_att)


        att_score_mat = att_score_mat.transpose(1,2).contiguous().view(att_score_mat.shape[0],-1,self.multi_head * self.h)

        return self.W(att_score_mat),self.attention_scores
