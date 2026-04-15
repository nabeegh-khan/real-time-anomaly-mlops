# LSTM Autoencoder architecture
# this file is imported by both the training notebook and the serving layer
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Encoder-decoder LSTM autoencoder for time series anomaly detection.

    Encoder: compresses the input sequence into a fixed-size hidden state
    Decoder: reconstructs the input sequence from that hidden state

    At inference time, reconstruction error (MSE) per window is the anomaly score.
    Anomalous windows produce high reconstruction error because the model
    only learned to reconstruct normal patterns during training.
    """

    def __init__(self, n_features, hidden_size, n_layers, dropout):
        super(LSTMAutoencoder, self).__init__()

        self.n_features  = n_features
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        # encoder: reads the input sequence and produces a context vector
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )

        # decoder: takes the context vector and reconstructs the sequence
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )

        # project decoder hidden states back to feature space
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        batch_size, seq_len, _ = x.shape

        # encode: run the full sequence, keep final hidden state
        _, (hidden, cell) = self.encoder(x)

        # decode: repeat context vector seq_len times as decoder input
        context = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(context, (hidden, cell))

        # project back to feature space
        reconstruction = self.output_layer(decoded)

        return reconstruction