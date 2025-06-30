import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from stylenet import StyleNet
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns


dataset_dir = 'cartoon_classification/TEST/'
weights_path = 'best_model_epoch50.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def get_embedding(model, img_path, device):
    img = load_image(img_path).to(device)
    with torch.no_grad():
        emb = model.forward_sample(img)
    return emb.squeeze(0).cpu().numpy()


model = StyleNet(device=device).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

embeddings = []
labels = []
label_names = []
img_paths = []
label2idx = {}

print("Extracting embeddings...")
for idx, class_name in enumerate(os.listdir(dataset_dir)):
    class_dir = os.path.join(dataset_dir, class_name)
    label2idx[class_name] = idx
    for fname in os.listdir(class_dir):
        if not fname.endswith('.jpg'):
            continue
        img_path = os.path.join(class_dir, fname)
        emb = get_embedding(model, img_path, device)
        embeddings.append(emb)
        labels.append(idx)
        label_names.append(class_name)
        img_paths.append(img_path)

embeddings = np.stack(embeddings)
labels = np.array(labels)
label_names = np.array(label_names)

print("Computing t-SNE...")
tsne = TSNE(n_components=2, random_state=0)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
tab20 = plt.get_cmap('tab20').colors
palette = [tab20[i % 20] for i in range(len(label2idx))]
for idx, class_name in enumerate(label2idx):
    inds = labels == idx
    plt.scatter(emb_2d[inds, 0], emb_2d[inds, 1], label=class_name, alpha=0.7, s=18, color=palette[idx])
plt.legend(fontsize=8)
plt.title("t-SNE of embeddings")
plt.tight_layout()
plt.savefig("tsne.png")
print("tsne plot saved!")

print("Computing intra/inter class distances...")
dist_matrix = euclidean_distances(embeddings, embeddings)
intra, inter = [], []
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        if labels[i] == labels[j]:
            intra.append(dist_matrix[i, j])
        else:
            inter.append(dist_matrix[i, j])

plt.figure(figsize=(8, 5))
sns.histplot(intra, color='blue', label='Intra-classe', kde=True, stat='density')
sns.histplot(inter, color='red', label='Inter-classe', kde=True, stat='density')
plt.legend()
plt.title('Euclidean distance')
plt.tight_layout()
plt.savefig("intra_class.png")
print("Intra class distance plot saved!")

print("Stats:")
print(f"Intra-class: avg {np.mean(intra):.4f}, std {np.std(intra):.4f}")
print(f"Inter-class: avg {np.mean(inter):.4f}, std {np.std(inter):.4f}")