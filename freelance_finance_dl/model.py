import torch
import torch.nn as nn


class TransactionAutoencoder(nn.Module):
    """
    LSTM sequence autoencoder for transaction anomaly detection.

    Encoder: reads the input sequence with an LSTM, projects the final
             hidden state down to a fixed-size latent vector.
    Decoder: expands the latent vector back to sequence length with an
             LSTM, then projects each timestep to a scalar output.

    Input/output shape: (batch, seq_len, 1)
    """

    def __init__(self, sequence_length=5, hidden_dim=64, latent_dim=16):
        super().__init__()

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_dim, 1)

    def encode(self, x):
        # x: (batch, seq_len, 1)
        _, (h_n, _) = self.encoder_lstm(x)
        # h_n: (1, batch, hidden_dim) -> squeeze to (batch, hidden_dim)
        latent = self.encoder_fc(h_n.squeeze(0))
        return latent  # (batch, latent_dim)

    def decode(self, latent):
        # Expand latent to (batch, seq_len, hidden_dim) as decoder input
        expanded = self.decoder_fc(latent)                          # (batch, hidden_dim)
        expanded = expanded.unsqueeze(1).repeat(1, self.sequence_length, 1)  # (batch, seq_len, hidden_dim)
        out, _ = self.decoder_lstm(expanded)                        # (batch, seq_len, hidden_dim)
        recon = self.output_fc(out)                                 # (batch, seq_len, 1)
        return recon

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)
