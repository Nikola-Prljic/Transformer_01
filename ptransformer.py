import torch
import torch.nn as nn

class ProcessTransformer(nn.Module):

    def __init__(self, vocab_size, PAD):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 256, padding_idx=PAD)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dropout=0.2,
                batch_first=True
            ),
            num_layers=4
        )

        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):

        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)

        return x
    
from torch.utils.data import Dataset

class ProcessDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        seq = self.sequences[idx]

        x = torch.tensor(seq[:-1])
        y = torch.tensor(seq[1:])

        return x, y