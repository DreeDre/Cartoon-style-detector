import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from cartoon_dataset import CartoonDataset


class StyleNet(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, 512)

    def forward_sample(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x, y, z):
        return self.forward_sample(x), self.forward_sample(y), self.forward_sample(z)


def triplet_loss(a, p, n, margin=1.0):
    pos_dist = F.pairwise_distance(a, p)
    neg_dist = F.pairwise_distance(a, n)
    loss = torch.mean(F.relu(pos_dist - neg_dist + margin))
    return loss


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = CartoonDataset(root_dir='cartoon_classification/TRAIN')
    val_dataset = CartoonDataset(root_dir='cartoon_classification/VAL')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = StyleNet(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=25)

    best_val_loss = float('inf')
    for epoch in tqdm(range(1, 51)):
        model.train()
        train_loss = 0
        train_batches = 0

        for anchor_img, positive_img, negative_img in tqdm(train_loader):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)

            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
        print(f"Epoch {epoch}, Loss: { train_loss / train_batches:.4f}")

        scheduler.step()
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            num_batches = 0
            with torch.no_grad():
                for anchor_img, positive_img, negative_img in tqdm(val_loader):
                    anchor_img = anchor_img.to(device)
                    positive_img = positive_img.to(device)
                    negative_img = negative_img.to(device)

                    anchor_emb, positive_emb, negative_emb = model(anchor_img, positive_img, negative_img)

                    loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
                    val_loss += loss.item()
                    num_batches += 1

            val_loss /= num_batches
            if  val_loss< best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_epoch{epoch}.pth')
                print(f'Best model saved! Validation Loss {best_val_loss:.4f}')
            else:
                print(f"Validation loss epoch {epoch}: {val_loss:.4f}")


if __name__ == "__main__":
    train()
