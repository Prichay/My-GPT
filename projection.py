from imports import *


class projectionLayer():
    
    def __init__(self, d_model, vocab_size) -> None:

        self.layer = torch.nn.Linear(d_model,vocab_size)
    
    def project(self,x):

        return torch.log_softmax(self.layer(x),dim=-1)