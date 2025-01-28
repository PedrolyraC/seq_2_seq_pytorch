from torch.onnx.symbolic_opset11 import unsqueeze
from treatment import *
from utils import *


class Encoder(nn.Module):
    """
    Seq2Seq model encoder. Would be responsible for receiving the german words and capture it's meaning to translate it
    in the decoder

    Parameters:
        input_size (int): size of the input sequence(german vocabulary)
        embedding_size (int): size of the embedding vector
        hidden_size (int): size of the hidden layer
        num_layers (int): number of LSTM layers in the encoder
        dropout_rate (float): dropout probability
    """

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_rate)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class Decoder(nn.Module):
    """
    Seq2Seq model decoder. Would be responsible for get the german words and translate it to english in the encoder

    Parameters:
        input_size (int): size of the input sequence(german vocabulary)
        embedding_size (int): size of the embedding vector
        hidden_size (int): size of the hidden layer
        output_size (int): size of the output sequence(english vocabulary)
        num_layers (int): number of LSTM layers in the encoder
        dropout_rate (float): dropout probability
    """

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_rate):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    """
    The translation model

    Parameters:
        encoder (Encoder): encoder model
        decoder (Decoder): decoder model
    """

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size. target_vocab_size).to(device)
        hidden, cell = self.encoder(source)
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs
