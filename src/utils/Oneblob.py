import torch

class One_Blob(torch.nn.Module):
   def __init__(self, lower_bound= -100.0, upper_bound=100.0, num_bins=1024,std_dev = 1.0):
        #one blob encoding encodes (1,B,3) tensor into (B,3*1024) tensor
        #one blob calculates the distance between the current bin and the kernel value
        #then assigns the kernel value to the bin according to a gaussian distribution
        super().__init__()
        self.gaussian = torch.distributions.normal.Normal(0,std_dev)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_bins = num_bins
        self.bin_size = (self.upper_bound - self.lower_bound) / self.num_bins
        self.bin_loc = torch.arange(self.lower_bound, self.upper_bound, self.bin_size)+self.bin_size/2
        self.bins = torch.arange(self.lower_bound, self.upper_bound, self.bin_size)
        self.gaussian.requires_grad = False

        # this function calculates the distance between the current bin and the kernel value

   def forward(self, x):
        
        # x is a (1,B,3) tensor
        x= x.squeeze(0)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        dis = self.bin_loc.repeat(3, 1)
        dis = dis.repeat(batch_size, 1, 1)
        # dis is a (B,3,1024) tensor
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, self.num_bins)
        dis = dis - x
        blob = self.gaussian.log_prob(dis)
        blob = torch.exp(blob)
        
        blob = blob.reshape(batch_size, -1)

        return blob




