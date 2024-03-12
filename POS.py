from imports import *


class POS_emb(torch.nn.Module):

    def __init__(self,tokenizer, emd_size):
        super(POS_emb, self).__init__()
        
        self.tokenizer = tokenizer
        self.emb_size = emd_size
        self.embed = torch.nn.Embedding(self.tokenizer.get_vocab_size(),self.emb_size)


    def positional_emb(self,dim,pos):
        
        return [math.sin(pos/10000**((2*ind)/dim)) if ind%2==0 else math.cos(pos/10000**((2*ind)/dim)) for ind in range(dim)]

    def generate(self,token_idx):

        _embedings      = [self.embed(torch.LongTensor(i.ids)) for i in token_idx]
        '''
        training Positional Embeddings for upto 50 sequence length (just in case)
        '''
        seq_length      = len(_embedings)
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