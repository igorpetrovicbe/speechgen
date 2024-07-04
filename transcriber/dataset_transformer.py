import numpy as np
import torch
import torchaudio
from natsort import natsorted
from pydub import AudioSegment
from torch.utils.data import Dataset
import os


def numpy_to_audio_segment(audio_data, sample_rate=16000):
    # Ensure the audio data is in the correct format (16-bit PCM)
    audio_data = (audio_data * 32767).astype(np.int16)

    # Create AudioSegment from NumPy array
    audio_segment = AudioSegment(
        audio_data.tobytes(),  # Convert NumPy array to bytes
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,  # Size of one sample in bytes
        channels=1  # Mono audio
    )

    return audio_segment


def audio_segment_to_numpy_array(audio_segment):
    # Get the raw samples as a NumPy array
    raw_samples = np.array(audio_segment.get_array_of_samples())

    # Normalize the samples to the range [-1, 1]
    normalized_samples = raw_samples / 32767.0

    return normalized_samples


class TranscriptionDataset(Dataset):
    def __init__(self, audio_root_dir, label_path, audio_sequence_length, text_sequence_length, device, vqvae,
                 vert_dim, hor_dim, hop_size, n_fft, num_mels, vocab_size=8192):
        self.label_path = label_path
        self.audio_sequence_length = audio_sequence_length
        self.vocab_size = vocab_size
        self.vqvae = vqvae
        self.device = device

        self.text_sequence_length = text_sequence_length

        self.hop_size = hop_size
        self.n_fft = n_fft
        self.num_mels = num_mels

        self.multiplier = 1

        audio_file_paths = [os.path.join(audio_root_dir, f) for f in os.listdir(audio_root_dir) if f.endswith('.wav') or
                            f.endswith('.mp3')]

        audio_file_paths = natsorted(audio_file_paths, key=lambda x: os.path.basename(x))

        self.audio_data = []
        self.indices_data = []
        self.label_data = []
        self.text_vocabulary = set()
        self.embedded_data = []

        self.first_run = True

        self.vert_dim = vert_dim
        self.hor_dim = hor_dim

        self.load_label_data(label_path)

        self.text_vocabulary = list('абвгдђежзијклљмнњопрстћуфхцчџш-.')

        self.char_to_index = {char: i for i, char in enumerate(self.text_vocabulary)}
        self.index_to_char = {i: char for i, char in enumerate(self.text_vocabulary)}

        counter = 0
        for file_name in audio_file_paths:
            sound_array = self.load_sound_file(file_name)
            self.audio_data.append(sound_array)
            print(counter)
            counter += 1

        for audio_clip in self.audio_data:
            mel_window = self.compute_mel_spectrogram(audio_clip, sample_rate=16000)
            indices = self.tokenize_item(mel_window)

            self.indices_data.append(indices)

    def __len__(self):
        return len(self.indices_data)

    def __getitem__(self, idx):
        audio_indices = self.indices_data[idx]
        label_data = self.label_data[idx // self.multiplier]
        # One-hot encode the sequences
        label = self.adjust_string(label_data)
        one_hot_label = self.string_to_one_hot(label)
        text_input = self.string_to_indices(label[:-1])
        one_hot_text_target = one_hot_label[1:]

        target_length = torch.tensor(len(label_data) - 2, dtype=torch.int32)

        if self.first_run:
            return torch.tensor(audio_indices), torch.tensor(text_input), one_hot_text_target, target_length
        else:
            return self.embedded_data[idx], torch.tensor(text_input), one_hot_text_target, target_length

    def load_label_data(self, file_path):
        # Open the file and read lines into a list
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Add '-' at the start and '.' at the end of each line, then remove newline characters
        self.label_data = ['-' + line.strip().replace(' ', '') + '.' for line in lines]  #

        # Reopen the file to read its content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Create a set of unique characters including '-' and '.'
        self.text_vocabulary = set(content.replace('\n', '').replace(' ', ''))  #.replace(' ', '')
        self.text_vocabulary.update({'-', '.'})

        # Print the list of lines and the set of unique characters
        print("List of lines:", self.label_data)
        print("Set of unique characters:", self.text_vocabulary)
        print(len(self.text_vocabulary))

    def load_sound_file(self, file_path):
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)

        print("Sample rate: " + str(sample_rate))

        # Convert waveform to mono if it has multiple channels
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad the waveform with zeros to the right if shorter than target_size
        if waveform.size(1) < 16000 * 8:
            padding_size = 16000 * 8 - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding_size), mode='constant', value=0)

        return waveform

    def tokenize_item(self, audio_mel):
        audio_mel = audio_mel.unsqueeze(0)
        indices_list = []

        with torch.no_grad():
            quantizer = self.vqvae.get_quantizer()
            encoding = self.vqvae.encode(audio_mel.to(self.device))
            encoding_indices = quantizer.get_quantized_indices(encoding)
            test_indices = encoding_indices.view(1, self.vert_dim, -1)
            test_indices2 = test_indices.transpose(1, 2)
            flat_indices = test_indices2.reshape(-1)
            indices_list.extend(flat_indices.tolist())

            indices_list.extend(encoding_indices.tolist())

        indices_list = indices_list[:self.audio_sequence_length]
        return torch.tensor(indices_list).to(self.device)

    def adjust_string(self, string):
        # Ensure that the string is not longer than the maximum length
        if len(string) > self.text_sequence_length + 1:
            string = string[:self.text_sequence_length + 1]
        # Pad the string with '.' if it is shorter than the maximum length
        else:
            string = string.ljust(self.text_sequence_length + 1, '.')

        return string

    def string_to_indices(self, string):
        # Tokenize string
        indices = []
        for i, char in enumerate(string):
            if char in self.char_to_index:
                indices.append(self.char_to_index[char])
        return indices

    def indices_to_string(self, indices):
        string = ''
        for i, index in enumerate(indices):
            if i in self.index_to_char:
                string += self.index_to_char[index]
        return string

    def string_to_one_hot(self, string):
        # Create a tensor to store the one-hot encoded sequence
        one_hot_tensor = torch.zeros(self.text_sequence_length + 1, len(self.text_vocabulary))

        # One-hot encode the string
        for i, char in enumerate(string):
            if char in self.char_to_index:
                one_hot_tensor[i, self.char_to_index[char]] = 1.0

        return one_hot_tensor

    def compute_mel_spectrogram(self, waveform, sample_rate):
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            n_mels=self.num_mels,
        )(waveform)

        # return mel_spectrogram
        return torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
