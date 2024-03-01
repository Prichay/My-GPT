from imports import *


class POS_emb():

    def __init__(self,tokenizer, token_id, emd_size):\
        
        self.tokenizer = tokenizer
        self.token_idx = token_id
        self.emb_size = emd_size

    def positional_emb(self,dim,pos):
        
        return [math.sin(pos/10000**((2*ind)/dim)) if ind%2==0 else math.cos(pos/10000**((2*ind)/dim)) for ind in range(dim)]

    def generate(self):

        embed           = torch.nn.Embedding(self.tokenizer.get_vocab_size(),self.emb_size)
        _embedings      = [embed(torch.LongTensor(i.ids)) for i in self.token_idx]
        '''
        training Positional Embeddings for upto 50 sequence length (just in case)
        '''
        seq_length      = 50
        '''
        create 512-dim embeddings for every single token 
        parameters : total vocab size from tokenizer
        so that like token came with "12652" as token_id then we have total vocab of 25000 which will make one-hot vector of 25000 with "1" at token_id
        '''

        pos_embeddings  = [self.positional_emb(512,i) for i in range(seq_length)]
        '''
        Embeddings + Pos_vector
        '''
        final_emb       = [_embedings[i]+torch.tensor(pos_embeddings[i]) for i,_ in enumerate(_embedings)]

        return final_emb