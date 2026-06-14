from torch.utils.data import DataLoader
from loadData import LoadData
from ptransformer import ProcessDataset, ProcessTransformer
import torch
import torch.nn as nn

PAD = 0

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    xs, ys = zip(*batch)

    xs = pad_sequence(list(xs), batch_first=True, padding_value=PAD)
    ys = pad_sequence(list(ys), batch_first=True, padding_value=PAD)
    return xs, ys

sequences, vocab = LoadData("./data/IC_variants.csv")
vocab_size = len(vocab) + 1

dataset = ProcessDataset(sequences)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

model = ProcessTransformer(vocab_size, PAD)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003
)

criterion = nn.CrossEntropyLoss()

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

for epoch in range(10):

    for x, y in loader:

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        print(epoch, loss.item())
        writer.add_scalar("Loss/train", loss, epoch)

writer.flush()
writer.close()

correct = 0
total = 0

with torch.no_grad():
    for x, y in loader:
        logits = model(x)

        pred = logits.argmax(dim=-1)

        mask = y != 0   # kein PAD zählen

        correct += (pred[mask] == y[mask]).sum().item()
        total += mask.sum().item()

accuracy = correct / total
print(accuracy)