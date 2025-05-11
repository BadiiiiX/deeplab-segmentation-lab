import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from project.loss import ComboLoss
from project.unet import UNet


class RadioMaskDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.radios = data['radios']  # (N, 8, 8)
        self.masks = data['masks']    # (N, H, W)

    def __len__(self):
        return len(self.radios)

    def __getitem__(self, idx):
        radio = torch.tensor(self.radios[idx], dtype=torch.float32).unsqueeze(0)  # (1, 8, 8)

        mask = torch.tensor(self.masks[idx], dtype=torch.float32)  # (H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)

        mask = F.interpolate(mask.unsqueeze(0), size=(64, 64), mode='nearest').squeeze(0)  # -> (1, 64, 64)

        return radio, mask


def show_sample(model, dataset, device, idx=0):
    model.eval()
    radio, mask = dataset[idx]
    radio, mask = radio.unsqueeze(0).to(device), mask.to(device)
    with torch.no_grad():
        pred = model(radio).squeeze().cpu()

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(radio.cpu().squeeze(), cmap='inferno')
    axs[0].set_title("Radio Input (8x8)")
    axs[1].imshow(mask.cpu().squeeze(), cmap='gray')
    axs[1].set_title("Ground Truth Mask (64x64)")

    pred_bin = (pred > 0.5).float()
    axs[2].imshow(pred_bin.numpy(), cmap='gray')
    #axs[2].imshow(pred, cmap='gray')

    axs[2].set_title("Predicted Mask (64x64)")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def train_model(dataset_path="result_10.npz", epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RadioMaskDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    loss_fn = ComboLoss(bce_weight=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for radios, masks in dataloader:
            radios, masks = radios.to(device), masks.to(device)
            preds = model(radios)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.4f}")

    return model, dataset, device


if __name__ == "__main__":
    model, dataset, device = train_model(dataset_path="../result_10.npz", epochs=10, batch_size=16)
    for idx in range(3):
        show_sample(model, dataset, device, idx=idx)
