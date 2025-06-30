import torch
from torchvision import transforms
from PIL import Image
import os
import random
from stylenet import StyleNet
from sklearn.metrics import accuracy_score, precision_score, recall_score


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
    return emb.squeeze(0).cpu()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_dir = 'cartoon_classification/TEST'
    classes = [cls for cls in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cls))]
    model = StyleNet(device=device).to(device)
    model.load_state_dict(torch.load('best_model_epoch.pth', map_location=device))
    model.eval()

    n_refs = 10
    threshold = 0.5
    n_iters = 10000

    y_true = []
    y_pred = []

    for i in range(n_iters):
        ref_label = random.choice(classes)
        ref_dir = os.path.join(dataset_dir, ref_label)
        ref_imgs = [os.path.join(ref_dir, fname) for fname in os.listdir(ref_dir) if fname.lower().endswith('.jpg')]
        ref_images_path = random.sample(ref_imgs, n_refs)

        query_label = random.choice(classes)
        query_dir = os.path.join(dataset_dir, query_label)
        query_imgs = [os.path.join(query_dir, fname) for fname in os.listdir(query_dir) if fname.lower().endswith('.jpg')]
        query_image_path = random.choice(query_imgs)

        ref_embeddings = [get_embedding(model, img_path, device) for img_path in ref_images_path]
        query_embedding = get_embedding(model, query_image_path, device)

        distances = [1 - torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), r.unsqueeze(0)).item()
                     for r in ref_embeddings]
        mean_distance = sum(distances) / len(distances)

        pred_same = mean_distance < threshold
        predicted_style = 1 if pred_same else 0
        real_style = 1 if query_label == ref_label else 0

        y_pred.append(predicted_style)
        y_true.append(real_style)

        if i % 50 == 0 and i > 0:
            acc = accuracy_score(y_true, y_pred)
            print(
                f"[{i}] Acc: {acc:.2%} | SAME STYLE precision: {precision_score(y_true, y_pred, pos_label=1):.2%} | "
                f"DIFFERENT STYLE precision: {precision_score(y_true, y_pred, pos_label=0):.2%}"
            )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, labels=[1,0])
    rec = recall_score(y_true, y_pred, average=None, labels=[1,0])

    print("\n---METRICS ---")
    print(f"Accuracy: {acc:.2%}")
    print(f"Precision (same style): {prec[0]:.2%}")
    print(f"Precision (different style): {prec[1]:.2%}")
    print(f"Recall (same style): {rec[0]:.2%}")
    print(f"Recall (different style): {rec[1]:.2%}")


if __name__ == "__main__":
    main()