from lib2to3.pgen2.tokenize import tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import embedding
from torch.onnx.symbolic_opset11 import unsqueeze
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
import treatment
from seq2seq_model import Decoder, Encoder, Seq2Seq

# Training Hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model Hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(treatment.german.vocab)
input_size_decoder = len(treatment.german.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (treatment.train_data, treatment.validation_data, treatment.test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key = lambda x: len(x.src),
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = treatment.english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}]')

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(model, sentence, german, english, device, max_len=50)

    for batch_idx, batch in enumerate(train_iterator):
        input_data = batch.src.to(device)
        target_data = batch.trg.to(device)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step +=1
