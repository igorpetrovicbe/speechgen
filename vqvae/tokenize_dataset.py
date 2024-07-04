# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.
import os

import librosa
import numpy as np
import torch
import torchaudio

from PIL import Image
from pydub import AudioSegment
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
#from vqvae_mel3 import VQVAE
from vqgan_mel import VQGAN
from dataset_h5 import MelSpectrogramDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

torch.set_printoptions(linewidth=160)


def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")


def save_spectrogram_images(spec_tensors, save_name=""):
    os.makedirs('spectrogram_images', exist_ok=True)

    for idx in range(spec_tensors.shape[0]):
        spec = spec_tensors[idx, 0].detach().cpu().numpy()

        plt.figure(figsize=(6, 4))
        plt.imshow(np.log2(spec), aspect='auto', cmap='viridis',
                   origin='lower')  # Using log scale for better visualization
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency Bin")

        save_path = os.path.join('spectrogram_images', f"{idx}_{save_name}.png")
        plt.savefig(save_path)
        plt.close()


window_size = 16000
hop_size = 252
num_mels = 128
n_fft = 1000

# Initialize model.
device = torch.device("cuda:0")
use_ema = True
model_args = {
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

if __name__ == '__main__':
    model = VQGAN(**model_args).to(device)

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    model.load_state_dict(torch.load('saved_state 4down 1hdown.pth')['model_state_dict'])

    file_path = 'dataset_v2.h5'

    print('h1')
    #train_dataset = MelSpectrogramDataset(path=file_path, window_size=window_size, hop_size=hop_size,
    #                                      num_mels=num_mels, n_fft=n_fft, transform=None, import_mode=True)
    train_dataset = MelSpectrogramDataset(path='', window_size=window_size, hop_size=hop_size,
                                          num_mels=num_mels, n_fft=n_fft, h5_filename='dataset_v2.h5', transform=None,
                                          import_mode=True)
    print('h2')
    # Initialize the DataLoader (replace this with your actual DataLoader initialization)
    unshuffled_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=12)

    indices_list = []

    counter = 0
    vert_dim = 8
    hor_dim = 32

    # Iterate over the DataLoader and sample N consecutive data points
    model.eval()
    with torch.no_grad():
        for batch in unshuffled_loader:
            # Assuming each batch contains both inputs and targets
            inputs = batch

            #outputs = model(inputs.to(device))["x_recon"]

            quantizer = model.get_quantizer()
            encoding = model.encode(inputs.to(device))
            encoding_indices = quantizer.get_quantized_indices(encoding)
            test_indices = encoding_indices.view(batch.shape[0], vert_dim, hor_dim)
            test_indices2 = test_indices.transpose(1, 2)
            flat_indices = test_indices2.reshape(-1)
            indices_list.extend(flat_indices.tolist())

            if (counter + 1) % 10 == 0:
                print(f'{counter}/{len(unshuffled_loader)}')
            counter += 1

    # Open a file in write mode
    with open(f'q_dataset.txt', 'w') as file:
        # Convert each integer to a string and join them with spaces
        content = '\n'.join(map(str, indices_list))

        # Write the content to the file
        file.write(content)
