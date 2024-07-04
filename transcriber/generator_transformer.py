import torch
import torch.nn as nn


class MultiLayerTransformer(nn.Module):
    def __init__(self, d_model, vocab_size, ff_size, num_layers, num_heads, max_sequence_length):
        super(MultiLayerTransformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length

        self.input_embedding = nn.Embedding(vocab_size, d_model)

        # Define the Transformer layer with multiple layers
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_size,
                                                      batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=num_layers)

        # Define the output layer
        self.out_fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #positional_encodings = self.create_positional_encoding(x.size(0), self.max_sequence_length, self.d_model)
        positional_encodings = self.create_positional_encoding(x.size(0), x.size(1), self.d_model)
        # Add positional encodings to the input
        x = self.input_embedding(x)
        #x = x + positional_encodings(torch.arange(x.size(1)).unsqueeze(0).to(x.device))
        x = x + positional_encodings[:, :self.max_sequence_length, :].to(x.device)

        # Generate causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Forward pass through the Transformer layer with attention mask
        transformer_out = self.transformer_encoder(x, mask=mask, is_causal=True)

        np_transformer_out = transformer_out.squeeze(0).cpu().numpy()

        # Forward pass through the output layer
        #output = self.out_fc(transformer_out)

        return transformer_out

    def forward_normal(self, x):
        #positional_encodings = self.create_positional_encoding(x.size(0), self.max_sequence_length, self.d_model)
        positional_encodings = self.create_positional_encoding(x.size(0), x.size(1), self.d_model)
        # Add positional encodings to the input
        x = self.input_embedding(x)
        #x = x + positional_encodings(torch.arange(x.size(1)).unsqueeze(0).to(x.device))
        x = x + positional_encodings[:, :self.max_sequence_length, :].to(x.device)

        # Generate causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Forward pass through the Transformer layer with attention mask
        transformer_out = self.transformer_encoder(x, mask=mask, is_causal=True)

        np_transformer_out = transformer_out.squeeze(0).cpu().numpy()

        # Forward pass through the output layer
        output = self.out_fc(transformer_out)

        return output

    def create_positional_encoding(self, batch_size, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc = torch.zeros((max_len, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # Add batch dimension
        return pos_enc
