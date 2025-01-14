from lib2to3.pgen2.tokenize import tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

# load tokenizers
spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def german_tokenizer(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def english_tokenizer(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# tokenize
german = Field(tokenize=german_tokenizer, lower=True,
               init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=english_tokenizer, lower=True,
               init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)
