import os
import torchaudio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import librosa
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from wandb.old.summary import h5py


class MelSpectrogramDataset(Dataset):
    def __init__(self, path, window_size, hop_size, num_mels, n_fft, h5_filename, transform=None, file_mode=False,
                 export_mode=False, import_mode=False):
        self.root_dir = path
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.transform = transform
        self.h5_filename = h5_filename
        self.current_index = 0

        if not import_mode:
            if file_mode:
                self.file_paths = [path]
            else:
                self.file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav') or f.endswith('.mp3')]

            with h5py.File(self.h5_filename, 'w') as h5_file:
                # Create a dataset placeholder for the mel spectrograms with an initial size
                self.create_h5_datasets(h5_file)

                for file_path in self.file_paths:
                    # Load the audio file
                    waveform, sample_rate = torchaudio.load(file_path)
                    print("Sample rate: " + str(sample_rate))

                    # Convert waveform to mono if it has multiple channels
                    if waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    file_len = waveform.size(1) // self.window_size
                    for i in range(file_len):
                        # Calculate the starting index for the current window
                        start_idx = i * self.window_size
                        # Extract the window from the waveform
                        waveform_window = waveform[:, start_idx:start_idx + self.window_size]
                        mel_window = self.compute_mel_spectrogram(waveform_window, sample_rate)
                        self.write_to_h5(h5_file, mel_window)

    def create_h5_datasets(self, h5_file):
        # Initial estimate of the number of windows; adjust as needed
        initial_windows = 1000  # Adjust this value as needed
        mel_shape = (initial_windows, self.num_mels, 64)  # Zakucano
        h5_file.create_dataset('mel_spectrograms', shape=(0,) + mel_shape[1:], maxshape=(None,) + mel_shape[1:], dtype='float32')

    def write_to_h5(self, h5_file, mel_window):
        mel_window = mel_window.squeeze(0)  # Ensure the mel_window has the correct dimensions
        if self.current_index >= h5_file['mel_spectrograms'].shape[0]:
            # Extend the dataset size
            h5_file['mel_spectrograms'].resize((h5_file['mel_spectrograms'].shape[0] + 1000), axis=0)
        h5_file['mel_spectrograms'][self.current_index] = mel_window.numpy()
        self.current_index += 1

    def __len__(self):
        with h5py.File(self.h5_filename, 'r') as h5_file:
            return len(h5_file['mel_spectrograms'])

    def __getitem__(self, idx):
        with h5py.File(self.h5_filename, 'r') as h5_file:
            mel_spectrogram = h5_file['mel_spectrograms'][idx]
        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)
        mel_spectrogram = torch.from_numpy(mel_spectrogram).unsqueeze(0)
        return mel_spectrogram

    def compute_mel_spectrogram(self, waveform, sample_rate):
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.num_mels,
        )(waveform)

        #return mel_spectrogram
        return torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    def extract_windows(self, mel_spec):
        # Extract small windows from the MEL spectrogram
        num_windows = mel_spec.size(2) // self.window_size
        windows = [mel_spec[:, :, i * self.window_size:(i + 1) * self.window_size] for i in range(num_windows)]
        return windows