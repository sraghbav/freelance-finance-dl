import torch.nn as nn


class TransactionAutoencoder(nn.Module):
    def __init__(self, sequence_length=5, hidden_dim=16, latent_dim=4):
        super().__init__()

        self.sequence_length = sequence_length

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.unsqueeze(-1)

        return decoded
