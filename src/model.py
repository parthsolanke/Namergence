import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Defines a simple RNN model.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.in_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in_to_out = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.in_to_hidden(combined)
        out = self.in_to_out(combined)
        out = self.softmax(out)
        return out, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)