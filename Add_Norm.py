from imports import *


class add_norm():

    def __init__(self,prev_layer_out,current_out,dropout) -> None:
        
        self.prev_output = prev_layer_out
        self.curr_output = current_out
        self.dropout = dropout
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))
        self.eps = 10**-6

    def Normalise(self):

        mean = self.curr_output.mean(dim =-1, keepdim = True)
        std = self.curr_output.std(dim =-1, keepdim = True)
    
        return (self.alpha * (self.curr_output - mean)/(std + self.eps)) + self.beta
    
    def res_net(self):

        return self.prev_output + self.Normalise()
    

