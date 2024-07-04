# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.
import os

import librosa
import numpy as np
import torch
import torchaudio
import wandb

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from vqgan_mel import VQGAN, Discriminator
#from dataset import MelSpectrogramDataset
from dataset_h5 import MelSpectrogramDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import wavfile

from torchaudio.transforms import GriffinLim
from pydub import AudioSegment

torch.set_printoptions(linewidth=160)


def save_img_tensors_as_grid(img_tensors, nrows, f):
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
        plt.imshow(spec, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Spectrogram {idx}")
        plt.xlabel("Time")
        plt.ylabel("Frequency Bin")

        save_path = os.path.join('spectrogram_images', f"{idx}_{save_name}.png")
        plt.savefig(save_path)
        plt.close()


def generate_audio():
    # Set the number of consecutive data points to sample
    N = 10

    # Initialize the DataLoader (replace this with your actual DataLoader initialization)
    unshuffled_loader = DataLoader(train_dataset, batch_size=N, shuffle=False)
    # Initialize an empty list to store consecutive chunks
    consecutive_chunks = []
    consecutive_chunks_lab = []

    # Iterate over the DataLoader and sample N consecutive data points
    model.eval()
    with torch.no_grad():
        for batch in unshuffled_loader:
            # Assuming each batch contains both inputs and targets
            inputs = batch

            outputs = model(inputs.to(device))["x_recon"]
            # Process the batch as needed (replace this with your actual processing logic)

            batch = batch
            outputs = outputs

            # Sample N consecutive data points
            for i in range(N):
                consecutive_chunk = outputs[i]
                consecutive_chunk_lab = batch[i]

                # Process the consecutive data point as needed
                consecutive_chunks.append(consecutive_chunk)
                consecutive_chunks_lab.append(consecutive_chunk_lab)

            break

    # Concatenate the consecutive chunks along the time dimension
    stacked_tensor = torch.cat(consecutive_chunks, dim=2).to('cpu')
    stacked_tensor_lab = torch.cat(consecutive_chunks_lab, dim=2).to('cpu')

    stacked_tensor = 10 ** (stacked_tensor / 10)
    stacked_tensor_lab = 10 ** (stacked_tensor_lab / 10)

    #stacked_tensor = stacked_tensor.squeeze(0)

    # Print the shape of the resulting stacked tensor
    print("Shape of the stacked tensor:", stacked_tensor.shape)

    save_spectrogram_images(stacked_tensor.unsqueeze(0), "bbb")
    save_spectrogram_images(stacked_tensor_lab.unsqueeze(0), "lab")

    sample_rate = 16000
    n_stft = int((n_fft // 2) + 1)

    inverse_transform = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_stft=n_stft, n_mels=num_mels)
    grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_size)

    inverse_waveform = inverse_transform(stacked_tensor)
    pseudo_waveform = grifflim_transform(inverse_waveform)

    torchaudio.save(f'reconstructed_audio_{iteration}.wav', pseudo_waveform, sample_rate)

    # LABEL

    inverse_waveform_lab = inverse_transform(stacked_tensor_lab)
    pseudo_waveform_lab = grifflim_transform(inverse_waveform_lab)

    torchaudio.save('reconstructed_audio_lab.wav', pseudo_waveform_lab, sample_rate)


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
    discriminator = Discriminator(model_args['in_channels'], model_args['num_hiddens']).to(device)

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    # Count the number of parameters
    num_discriminator_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Number of parameters in the discriminator: {num_discriminator_params}")

    # Initialize dataset.
    batch_size = 32
    workers = 1  # 4
    normalize = transforms.Normalize(mean=0.5, std=1.0)
    transform = transforms.Compose(
        [
            # normalize,
        ]
    )

    #data_root = 'dataset'
    #data_root = 'H:\\PycharmProjects\\vqvae-audiotrack\\srpski_train_dataset'
    data_root = 'H:\\PycharmProjects\\VoiceAugmenter\\output'
    #dataset_path = 'dataset_v1.h5'

    # data_root = "../data"
    #full_dataset = MelSpectrogramDataset(path=dataset_path, window_size=window_size, hop_size=hop_size,
    #                                     num_mels=num_mels, n_fft=n_fft, transform=None, import_mode=True)
    full_dataset = MelSpectrogramDataset(path=data_root, window_size=window_size, hop_size=hop_size,
                                         num_mels=num_mels, n_fft=n_fft, h5_filename='dataset_v2.h5', transform=None, import_mode=True)


    train_dataset = full_dataset

    # Calculate the variance of the transformed data with a progress bar
    variances = []
    for img in tqdm(train_dataset, desc="Calculating variance", dynamic_ncols=True):
        variances.append(np.var(np.array(img)))

    # Compute the overall variance
    train_data_variance = np.var(np.array(variances))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation
    # Learning".
    beta = 0.25

    # Warm-up parameters
    use_warmup = False
    warmup_iterations = 2000

    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    discriminator_params = [params for params in discriminator.parameters()]
    lr = 0.0003
    discriminator_lr = 0.0003
    initial_lr = 1e-5
    lr_increment = (lr - initial_lr) / warmup_iterations  # Incremental change in learning rate per iteration

    lambda_discriminator = 0  # 0.01

    if use_warmup:
        optimizer = optim.Adam(train_params, lr=initial_lr)
    else:
        optimizer = optim.Adam(train_params, lr=lr)

    discriminator_optimizer = optim.Adam(discriminator_params, lr=discriminator_lr)
    criterion = nn.MSELoss()
    criterion_discriminator = nn.BCELoss()

    iteration = 0
    running_loss = 0

    # Initialize other variables needed for saving
    saved_state = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="serbian-mel-audio-vqgan",

        name=f"{model_args['num_residual_hiddens']}h {model_args['num_downsampling_layers']}down "
             f"{model_args['num_hor_downsampling_layers']}hdown {batch_size}b {lambda_discriminator}lmbd",

        # track hyperparameters and run metadata
        config={
            "learning_rate_start": lr,
            "hidden_size": model_args['num_hiddens'],
            "num_downsampling_layers": model_args['num_downsampling_layers'],
            "num_residual_layers": model_args['num_residual_layers'],
            "batch_size": batch_size,
            "num_hor_downsampling_layers": model_args['num_hor_downsampling_layers'],
            "dataset": "v2",
        }
    )

    generate_every = 2000
    save_every = 1700
    export_images_every = 500 #5000

    # Train model.
    epochs = 100
    eval_every = 5
    best_train_loss = float("inf")
    for epoch in range(epochs):
        total_train_loss = 0
        total_recon_error = 0
        total_discriminator_loss = 0
        total_vq_discriminator_loss = 0
        total_accuracy = 0
        n_train = 0
        for (batch_idx, train_tensors) in enumerate(train_loader):
            model.train()
            discriminator.train()
            optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            imgs = train_tensors.to(device)
            out = model(imgs)
            recon_error = criterion(out["x_recon"], imgs) / train_data_variance
            total_recon_error += recon_error.item()

            # Compute VQ-VAE loss
            vq_vae_loss = recon_error + beta * out["commitment_loss"]
            if not use_ema:
                vq_vae_loss += out["dictionary_loss"]

            with torch.no_grad():
                real_preds = discriminator(imgs)
                fake_preds = discriminator(out["x_recon"].detach())

                # Concatenate predictions and labels
                all_preds = torch.cat([real_preds, fake_preds], dim=0)
                all_labels = torch.cat([torch.ones_like(real_preds), torch.zeros_like(fake_preds)], dim=0)

            accuracy = ((all_preds >= 0.5) == all_labels).float().mean().item()

            total_accuracy += accuracy

            # Compute discriminator loss
            real_labels = torch.ones(imgs.size(0), 1, device=device)
            fake_labels = torch.zeros(imgs.size(0), 1, device=device)
            real_loss = criterion_discriminator(discriminator(imgs), real_labels)
            fake_loss = criterion_discriminator(discriminator(out["x_recon"].detach()), fake_labels)
            discriminator_loss = (real_loss + fake_loss) / 2
            total_discriminator_loss += discriminator_loss.item()

            # Compute distriminator loss for VQ VAE
            if accuracy > 0.5:
                #vq_real_labels = torch.zeros(imgs.size(0), 1, device=device)
                vq_fake_labels = torch.ones(imgs.size(0), 1, device=device)
                #vq_real_loss = criterion_discriminator(discriminator(imgs), vq_real_labels)
                vq_fake_loss = criterion_discriminator(discriminator(out["x_recon"]), vq_fake_labels)
                vq_discriminator_loss = vq_fake_loss * lambda_discriminator
                total_vq_discriminator_loss += vq_discriminator_loss.item()

                # Add discriminator loss penalty to VQ-VAE loss and subtract from discriminator loss
                #vq_vae_loss = vq_discriminator_loss
                #vq_vae_loss += vq_discriminator_loss

                total_train_loss += vq_vae_loss.item()

                vq_vae_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Learning rate warm-up
            if use_warmup:
                if iteration <= warmup_iterations:
                    new_lr = initial_lr + lr_increment * iteration
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

            n_train += 1

            if ((batch_idx + 1) % eval_every) == 0:
                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}\nlr: {optimizer.param_groups[0]['lr']}", flush=True)
                total_train_loss /= n_train
                total_discriminator_loss /= n_train
                total_recon_error /= n_train
                total_vq_discriminator_loss /= n_train
                total_accuracy /= n_train
                if total_train_loss < best_train_loss:
                    best_train_loss = total_train_loss

                print(f"total_train_loss: {total_train_loss}")
                print(f"best_train_loss: {best_train_loss}")
                print(f"recon_error: {total_recon_error}")
                print(f"discriminator_loss: {total_discriminator_loss}\n")
                print(f"VQ VAE discriminator_loss: {total_vq_discriminator_loss}\n")
                print(f"Discriminator Accuracy: {total_accuracy}\n")

                total_train_loss = 0
                total_recon_error = 0
                total_vq_discriminator_loss = 0
                total_accuracy = 0

                total_discriminator_loss = 0
                n_train = 0

            if (iteration + 1) % generate_every == 0:
                generate_audio()

            if (iteration + 1) % save_every == 0:
                saved_state['iteration'] = iteration
                saved_state['model_state_dict'] = model.state_dict()
                saved_state['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(saved_state, f'saved_state.pth')

            wandb.log({"loss": recon_error.item()})
            iteration += 1

            # Generate and save reconstructions.
            if (iteration + 1) % export_images_every == 0:
                model.eval()

                valid_dataset = train_dataset #CIFAR10(data_root, False, transform, download=True)
                valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=4,
                )

                with torch.no_grad():
                    for valid_tensors in valid_loader:
                        break
                    save_spectrogram_images(valid_tensors, "true")
                    save_spectrogram_images(model(valid_tensors.to(device))["x_recon"], "recon")

                generate_audio()
