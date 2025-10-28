import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from ddsp_guitar.model import GuitarDDSP
from ddsp_guitar.losses.msstft import MultiScaleSTFTLoss
from ddsp_guitar.losses.transient import TransientLoss


class DummyDataset(Dataset):
    def __init__(self, n=100, length=48000):
        super().__init__()
        self.n = n
        self.length = length

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randn(self.length)
        f0 = torch.full((self.length // 256 + 1,), 110.0)
        loud = torch.zeros_like(f0)
        y = torch.tanh(x)
        return x, f0, loud, y


def main():
    ds = DummyDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    model = GuitarDDSP()
    stft = MultiScaleSTFTLoss()
    l1 = nn.L1Loss()
    trans = TransientLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for it, (x, f0, loud, y) in enumerate(dl):
        y_pred = model(x, f0, loud)
        loss = stft(y_pred, y) + 0.5 * l1(y_pred, y) + 0.5 * trans(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(it, float(loss))
        if it > 3:
            break


if __name__ == "__main__":
    main()
