import spacy
import torch
from torchtext.data.metrics import bleu_score

sentence = 'ein boot mit mehreren mannern darauf wird von einem groben pferdegespann ans ufer gezogen.'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def translate_sentence(model, used_sentence, german, english, used_device, max_length=50):
    """
    Make the transalation of a sentence
    :param model:
    :param used_sentence:
    :param german:
    :param english:
    :param used_device:
    :param max_length:
    :return:
    """
    # Load german tokenizer
    spacy_ger = spacy.load("de")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(used_sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(used_sentence)]
    else:
        tokens = [token.lower() for token in used_sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(used_device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(used_device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]

def bleu(data, model, german, english, used_device):
    """
    Calculates the BLEU(Bilingual Evaluation Understudy) score for a given sentence

    :param data:
    :param model:
    :param german:
    :param english:
    :param used_device:
    :return:
    """
    targets = []
    outputs = []

    for example in data:
        source = vars(example)['src']
        target = vars(example)['tgt']

        prediction = translate_sentence(model, source, german, english, used_device)
        prediction = prediction[:-1] #remove <eos> token

        targets.append(target)
        outputs.append(prediction)

    return bleu_score

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    """
        Saves the checkpoint
        :param state:
        :param :
        """
    print(f'Saving checkpoint to {filename}')
    torch.save(state, filename)
    print('Checkpoint saved')

def load_checkpoint(checkpoint, model, optimizer):
    """
    Loads the checkpoint
    :param checkpoint:
    :param model:
    :param optimizer:
    """
    print(f'Loading checkpoint from {checkpoint}')
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def bleu_score_test(test_data, model, german, english, used_device):
    """
    Make the BLEU score test
    :param test_data:
    :param model:
    :param german:
    :param english:
    :return:
    """
    score = bleu(test_data[1:100], model, german, english, used_device)
    return f'Bleu score: {score*100:.2f}'
