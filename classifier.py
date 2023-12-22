
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NappaSleepNet(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=10,
                 num_layers=2, bidirectional=True, padding_value=-1):
        """
        Initialize the NappaSleepNet model for classifying sleep stages.

        Args:
          n_features (int): The number of features for each time step in the input.
          n_classes (int): The number of possible classes (sleep stages) for classification.
          hidden_size (int): The number of dimensions in the hidden state
          num_layers (int): The number of recurrent layers in the GRU.
          bidirectional (bool): If True, becomes a bidirectional GRU.
          padding_value (int): The padding value used to fill the tensor to match the longest 
          sleep sequence in each batch during training of the classifier.
        """
        super(NappaSleepNet, self).__init__()

        self.padding_value = padding_value
        
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size,
                          batch_first=True, bidirectional=bidirectional, num_layers=num_layers)

        # Define the output layer that maps the hidden state to class logits.
        # If bidirectional, the hidden size is doubled as it concatenates the features from both directions.
        self.out = nn.Linear(in_features=hidden_size*2 if bidirectional else hidden_size, 
                             out_features=n_classes)

    def forward(self, x, rec_lengths):
        """
        Forward pass through the network.
        
        Args:
          x (Tensor): The input tensor containing features of shape (batch_size, rec_length, n_features).
          rec_lengths (Tensor): The actual lengths of each sleep sequence before padding.

        Returns:
          Tensor: The output logits of shape (batch_size, rec_length, n_classes).
        """
        # Pack the padded sequence to remove the padding effect during GRU processing.
        x = pack_padded_sequence(x, rec_lengths, batch_first=True, enforce_sorted=False)
        
        x, _ = self.gru(x)
        
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=self.padding_value)
        
        # Pass the output of the GRU through the output layer to get class logits.
        x = self.out(x)
        
        return x

    def reset(self):
        """
        Reset the parameters of the model. This is called before training a new model instance in LOSOCV.
        """
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
