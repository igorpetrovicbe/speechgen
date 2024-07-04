import random

import torch
import torchaudio
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
from transformer_full_indices import MultiLayerTransformer
from dataset_audio_full_indices_general import AudioDataset
import os
import numpy as np
import itertools

import wandb

from vqgan_mel import VQGAN

vocab_size = 8192
chunk_size = 16000
sequence_length = 2048

hop_size = 252
num_mels = 128
n_fft = 1000

# Set the seed for PyTorch
seed = 42
torch.manual_seed(seed)

# Set the seed for NumPy if you are using NumPy functions
np.random.seed(seed)


def embed_indices(vqvae, indices):
    with torch.no_grad():
        quantizer = vqvae.get_quantizer()
        indices = torch.tensor(indices)
        #test_indices = indices.view(indices.shape[0], 32, 2)
        #test_indices2 = test_indices.transpose(1, 2)
        #corrected_indices = test_indices2.reshape(1, -1)
        x_tensor = torch.zeros((1, 256, vert_size, indices.shape[1] // vert_size))
        input_embedding = quantizer.quantize_indices(x_tensor, indices)
        input_embedding = input_embedding.squeeze(0).permute(1, 2, 0)
        input_embedding = input_embedding.view(1, input_embedding.shape[1] * vert_size, input_embedding.shape[2])
    return input_embedding


def generate_integer_sequence(model, vqvae, device, seed_sequence, length=100, temperature=1.0):
    model.eval()  # Set the model to evaluation mode
    generated_sequence = seed_sequence.copy()

    step = 0
    with torch.no_grad():
        for _ in range(length):
            # Convert the seed sequence to tensor
            seed_tensor = torch.tensor(seed_sequence).unsqueeze(0).to(device)

            # Forward pass
            output = model(seed_tensor)

            output_np = output.cpu().numpy()

            output = output[:, -1, :]

            output_np_last = output.cpu().numpy()

            # Apply temperature to the output to control randomness
            output = output.squeeze(0) / temperature

            output_np_last_temped = output.cpu().numpy()

            probabilities = torch.nn.functional.softmax(output, dim=0)

            probabilities_np = probabilities.cpu().numpy()

            # Sample the next index based on the probabilities
            next_index = torch.multinomial(probabilities, 1).item()

            # Add the next index to the generated sequence
            generated_sequence.append(next_index)

            # Update the seed sequence for the next iteration
            if len(seed_sequence) >= sequence_length:
                seed_sequence = seed_sequence[vert_size:] + [next_index]
            else:
                seed_sequence = seed_sequence[:] + [next_index]

            if (step + 1) % 100 == 0:
                print(f'{step}/{length}')

            step += 1

    return generated_sequence


# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

load_mode = False

if load_mode:
    loaded_state = torch.load('saved_state_iteration_139999.pth')

# Initialize model.
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

vqvae = VQGAN(**vqvae_args).to(device)

batch_size = 16
accumulated_batch_size = 256
gradient_accumulation_steps = int(accumulated_batch_size / batch_size)  #int(1000 * 32 / batch_size)
input_size = 512
ff_size = 4 * input_size
output_size = vocab_size
num_layers = 6
num_heads = int(input_size / 64)
learning_rate = 0.001
learning_rate_end = learning_rate * 0.1
num_epochs = 100
weight_decay = 0.1

vert_size = 8

# Set the warm-up parameters
use_warmup = True
warmup_iters = 2000 * gradient_accumulation_steps
warmup_init_lr = learning_rate * 0.1  # 1e-5

# Create dataset and data loader
current_directory = os.getcwd()
dataset_path = os.path.join(current_directory, 'Serbian Mel dataset 4down 1hdown')
print(dataset_path)
dataset = AudioDataset(root_dir=dataset_path, sequence_length=sequence_length, device=device, vert_size=vert_size,
                       vocab_size=vocab_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Hyperparameters
#input_size = dataset.vocab_size

# Create an instance of the SimpleRNN model
#model = MultiLayerRNN(input_size, hidden_size, output_size, num_layers)
#model = MultiLayerLSTM(input_size, hidden_size, output_size, num_layers)
model = MultiLayerTransformer(input_size, vocab_size, ff_size, num_layers, num_heads, sequence_length)
if load_mode:
    model.load_state_dict(loaded_state['model_state_dict'])

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

# Count the number of parameters in VQ VAE
num_params = sum(p.numel() for p in vqvae.parameters())
print(f"Number of parameters in the VQ VAE: {num_params}")

vqvae.load_state_dict(torch.load('vqgan_4down_1hdown.pth')['model_state_dict'])

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = AdamW(model.parameters(), lr=warmup_init_lr, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-6)
if load_mode:
    optimizer.load_state_dict(loaded_state['optimizer_state_dict'])

#max_steps = total_params * 20 / batch_size
max_steps = 600000 * 480 / (batch_size * gradient_accumulation_steps) * (total_params / 124000000)

# Add a cosine annealing learning rate scheduler
max_gradient_norm = 1.0
scheduler = CosineAnnealingLR(optimizer, T_max=int(max_steps), eta_min=learning_rate_end)
if load_mode:
    scheduler.load_state_dict(loaded_state['scheduler_state_dict'])

# Set the value of K for plotting every K steps
plot_every_k_steps = int(1000 * 32 / batch_size)  # gradient_accumulation_steps
generate_frequency = int(1000 * 32 / batch_size)  #500000
save_every_n_iterations = 10000
losses = []  # To store the training losses
running_loss = 0.0  # To accumulate the loss over K steps

iteration = 0
start_batch_idx = 0
if load_mode:
    iteration = loaded_state['iteration'] + 1
    start_batch_idx = loaded_state['batch_idx'] + 1
    running_loss = 0.0

    # Set the learning rate in the optimizer based on the current iteration
    if iteration < warmup_iters:
        lr = warmup_init_lr + (learning_rate - warmup_init_lr) * (iteration / warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print('Preskip point')
    skipped_batches = itertools.islice(data_loader, start_batch_idx, None)
    print('Skipped1')
    data_loader.__iter__ = skipped_batches.__iter__
    print('Skipped2')

# Initialize other variables needed for saving
saved_state = {
    'iteration': iteration,
    'batch_idx': start_batch_idx,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'running_loss': running_loss,
}

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="serbian-mel-speech-gen",

    name=f"Transformer Full {input_size}d {num_layers}l {num_heads}heads",

    # track hyperparameters and run metadata
    config={
        "learning_rate_start": learning_rate,
        "learning_rate_end": learning_rate_end,
        "context_length": sequence_length,
        "ff_size": ff_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size * gradient_accumulation_steps,
        "warmup_iters": warmup_iters,
        "warmup_init_lr": warmup_init_lr,
        "dataset": "v1",
    },
)
#    resume="must",
#    id="whmjtdf4"
total_loss = 0

if __name__ == "__main__":
    for epoch in range(num_epochs):
        for step, (inputs, targets) in enumerate(data_loader):
            model.train()  # Set the model to training mode
            # Move inputs and targets to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Update learning rate during warm-up phase
            if use_warmup and iteration < warmup_iters:
                lr = warmup_init_lr + (learning_rate - warmup_init_lr) * (iteration / warmup_iters)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Forward pass
            outputs = model(inputs)
            #log_probs = F.log_softmax(outputs, dim=-1)

            # Compute the loss
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1, vocab_size))
            #loss = criterion(log_probs, targets)

            # Check for NaN or Inf loss
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                print(f"NaN or Inf loss encountered at epoch {epoch + 1}, step {step + 1}. Skipping update step.")
                continue

            # Backward pass and optimization
            loss.backward()

            total_loss += loss.item()

            if (iteration + 1) % gradient_accumulation_steps == 0:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

                optimizer.step()
                optimizer.zero_grad()

                print('Batch Loss: ' + str(total_loss / gradient_accumulation_steps) + ', lr:' +
                      str(optimizer.param_groups[0]['lr']))
                total_loss = 0.0

            # Update the learning rate
            scheduler.step()

            # Accumulate the loss
            running_loss += loss.item()

            # Print loss every K steps
            if (iteration + 1) % plot_every_k_steps == 0:
                average_loss = running_loss / plot_every_k_steps
                current_lr = scheduler.get_lr()[0]  # Get the current learning rate
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(data_loader)}], Average Loss: {average_loss}'
                      f', Learning Rate: {current_lr}')

                print(f'Iteration: {iteration}/{max_steps}')

                # Store the average loss for plotting
                losses.append(average_loss)

                wandb.log({"loss": average_loss})

                # Reset running_loss for the next K steps
                running_loss = 0.0

            # Plot the training loss every K steps
            if (iteration + 1) % plot_every_k_steps == 0:
                print('Plot updated')
                #plt.close()
                #plt.plot(losses, label='Average Training Loss')
                #plt.xlabel('Step (every {} steps)'.format(plot_every_k_steps))
                #plt.ylabel('Loss')
                #plt.title('Average Training Loss Over Steps')
                #plt.legend()
                #plt.ylim(top=0.35, bottom=0.0)

                #plt.savefig('loss.png')
                #plt.show(block=False)
                #plt.pause(0.01)

            if (iteration + 1) % generate_frequency == 0 or iteration + 1 == max_steps:
                model.eval()
                with torch.no_grad():
                    # Choose a random seed sequence from the dataset
                    random_idx = random.randint(0, len(dataset) - 1)
                    seed_sequence = dataset[random_idx][0]  # Take the input sequence from the dataset

                    total_length = sequence_length + 256 * 4

                    seed_list = seed_sequence.tolist()

                    # Generate text using the trained RNN
                    generated_indices = generate_integer_sequence(model, vqvae, device, seed_list,
                                                                  length=total_length - sequence_length, temperature=1.0)

                    generated_indices = torch.tensor(generated_indices).to(device)

                    test_indices = generated_indices.view(-1, total_length // vert_size, vert_size)
                    test_indices2 = test_indices.transpose(1, 2)
                    corrected_indices = test_indices2.reshape(-1)

                    quantizer = vqvae.get_quantizer()
                    x_tensor = torch.zeros((1, vqvae_args['embedding_dim'], vert_size, total_length // vert_size))  # 2 x 128
                    quantized_x = quantizer.quantize_indices(x_tensor, corrected_indices)
                    outputs = vqvae.decode(quantized_x).to('cpu').squeeze(0)

                    outputs_linear = 10 ** (outputs / 10)

                    sample_rate = 16000
                    n_stft = int((n_fft // 2) + 1)

                    inverse_transform = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_stft=n_stft,
                                                                              n_mels=num_mels)
                    grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_size)

                    inverse_waveform = inverse_transform(outputs_linear)
                    pseudo_waveform = grifflim_transform(inverse_waveform)

                    torchaudio.save(f'output_{iteration}.wav', pseudo_waveform, sample_rate)

                    # Print the generated indices
                    print(f"Generated {len(generated_indices)} indices:")
                    print(generated_indices)

            if (iteration + 1) % save_every_n_iterations == 0 or iteration + 1 == max_steps:
                saved_state['iteration'] = iteration
                saved_state['batch_idx'] = step
                saved_state['model_state_dict'] = model.state_dict()
                saved_state['optimizer_state_dict'] = optimizer.state_dict()
                saved_state['scheduler_state_dict'] = scheduler.state_dict()
                saved_state['running_loss'] = running_loss

                torch.save(saved_state, f'saved_state_iteration_{iteration}.pth')

            iteration += 1


    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
