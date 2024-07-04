import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset_transformer import TranscriptionDataset
from vqgan_mel import VQGAN
from generator_transformer import MultiLayerTransformer
from transcriber_cnn_timed import TranscriptionCNN

from fuzzywuzzy import fuzz

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

test_mode = True

audio_sequence_length = 2048
text_sequence_length = 160
vocab_size = 8192
device = torch.device("cuda:0")
chunk_size = 16000
hop_size = 252
num_mels = 128
n_fft = 1000

input_size = 768
ff_size = 4 * input_size
output_size = vocab_size
num_layers = 12
num_heads = int(input_size / 64)

# Set the seed for PyTorch
seed = 42
torch.manual_seed(seed)

# Set the seed for NumPy if you are using NumPy functions
np.random.seed(seed)

use_ema = True
vqvae_args = {
    "in_channels": 1,
    "num_mels": num_mels,
    "num_hiddens": 256,
    "num_downsampling_layers": 4,
    "num_hor_downsampling_layers": 1,
    "num_residual_layers": 2,
    "num_residual_hiddens": 256,
    "embedding_dim": 256,
    "num_embeddings": 8192,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}

vert_dim = 8
hor_dim = 2

def embed_indices(vqvae, indices):
    with torch.no_grad():
        quantizer = vqvae.get_quantizer()
        indices = torch.tensor(indices)
        x_tensor = torch.zeros((1, 256, vert_dim, indices.shape[1] // vert_dim))
        input_embedding = quantizer.quantize_indices(x_tensor, indices)
        input_embedding = input_embedding.squeeze(0).permute(1, 2, 0)
        input_embedding = input_embedding.view(1, input_embedding.shape[1] * vert_dim, input_embedding.shape[2])
    return input_embedding


def decode_transcriber_output(generator_representation, transcriber, dataset):
    # Apply the linear layer
    logits = transcriber(generator_representation)  # Shape: (batch_size, sequence_length, num_classes)

    # Here, we use argmax to get the most probable token indices
    predicted_indices = torch.argmax(logits, dim=-1)  # Shape: (batch_size, sequence_length)

    # Convert indices to tokens
    predicted_tokens = []
    for indices in predicted_indices:
        #tokens = [vocab[idx.item()] for idx in indices]
        tokens = [dataset.index_to_char[idx.item()] for idx in indices]
        predicted_tokens.append(tokens)

    # Join tokens to form text
    predicted_text = ["".join(tokens) for tokens in predicted_tokens]

    return predicted_text[0]


def process_text(ctc_output, blank_char='.'):
    processed_output = []
    previous_char = None

    for char in ctc_output:
        if char != blank_char and char != previous_char:
            processed_output.append(char)
        if char != blank_char:
            previous_char = char

    return "".join(processed_output)


def evaluate_performance():
    total_ratio = 0
    counter = 0
    transcriber.eval()
    valid_loader2 = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    try:
        with torch.no_grad():
            for i, batch in enumerate(valid_loader2):
                # Assuming each batch contains both inputs and targets
                audio_input, text_input, text_target, _ = batch

                audio_input = audio_input.to(device)

                with torch.no_grad():
                    generator_representation = generator(audio_input)[:, 7::8, :]

                generator_representation = generator_representation.to(device)
                text_input = text_input.to(device)
                text_target = text_target.to(device)

                output_text = decode_transcriber_output(generator_representation, transcriber, valid_dataset)
                print(output_text)
                processed_text = process_text(output_text)
                print(processed_text)
                print(correct_labels[i])
                ratio = fuzz.ratio(processed_text, correct_labels[i])
                total_ratio += ratio
                print(f'Ratio: {ratio}')
                counter += 1
    except:
        print('End of labels error')
    print(f'Total Ratio: {total_ratio / counter}')
    return total_ratio / counter

vqvae = VQGAN(**vqvae_args).to(device)
vqvae.load_state_dict(torch.load('weights/vqgan_4down_1hdown.pth'))

generator = MultiLayerTransformer(input_size, vocab_size, ff_size, num_layers, num_heads, audio_sequence_length).to(device)
generator.load_state_dict(torch.load('weights/generator_transformer_100M.pth'))


transcriber_input_size = input_size
transcriber_output_size = 32

filter_count = 128
res_blocks = 2

transcriber = TranscriptionCNN(vocab_size=transcriber_output_size, d_model=input_size, filter_count=filter_count,
                               num_residual_blocks=res_blocks, device=device).to(device)

transcriber.load_state_dict(torch.load('weights/transcriber.pth'))

total_params = sum(p.numel() for p in transcriber.parameters())
print(f"Total number of parameters in the transcriber: {total_params}")

vqvae.eval()
generator.eval()
transcriber.eval()

if not test_mode:
    valid_label_path = 'data/audio_labels_valid.txt'
    valid_audio_root = 'data/audioclips_valid_set'
else:
    valid_label_path = 'data/audio_labels_test.txt'
    valid_audio_root = 'data/audioclips_test_set'

print('---------Valid-----------')
valid_dataset = TranscriptionDataset(audio_root_dir=valid_audio_root, label_path=valid_label_path,
                audio_sequence_length=audio_sequence_length, text_sequence_length=text_sequence_length,
                device=device, vqvae=vqvae, vert_dim=vert_dim, hor_dim=hor_dim, hop_size=hop_size,
                n_fft=n_fft, num_mels=num_mels, vocab_size=vocab_size)


valid_dataset.multiplier = 1

with open(valid_label_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
correct_labels = [process_text(line.strip(), blank_char=' ') for line in lines]

list_ratios = []

_ = evaluate_performance()
