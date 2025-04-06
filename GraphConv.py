import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, BatchNorm, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import re
import numpy as np
from scipy.spatial import KDTree

# ✅ Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1️⃣ **Convert Image to Graph**
class ImageToGraph:
    def __init__(self, img_size=64):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        self.img_size = img_size

    def convert(self, img_path, label):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        h, w = self.img_size, self.img_size

        # ✅ **Node Features (RGB Pixel Values Normalized)**
        x = img_tensor.permute(1, 2, 0).reshape(-1, 3)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)  # Normalize

        # ✅ **k-NN Graph Construction (Better than Grid)**
        coords = [(i // w, i % w) for i in range(h * w)]
        tree = KDTree(coords)
        edge_index = []
        for i in range(h * w):
            _, neighbors = tree.query(coords[i], k=5)
            for j in neighbors:
                if i != j:
                    edge_index.append([i, j])

        return Data(
            x=x.to(torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            y=torch.tensor([label], dtype=torch.long)
        )

# 2️⃣ **Dataset Loader**
class CaptchaDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.converter = ImageToGraph()

        # ✅ **Load image files**
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(os.path.join(root, ext)))

        if not self.image_files:
            raise ValueError(f"No images found in {root}")

        # ✅ **Extract labels from filenames**
        self.labels = []
        for f in self.image_files:
            num = re.sub(r'\D', '', os.path.basename(f).split('.')[0])
            label = int(num) if num else 0
            self.labels.append(label)

        # ✅ **Get feature dimensions from first sample**
        sample_data = self.converter.convert(self.image_files[0], self.labels[0])
        self._num_features = sample_data.x.shape[1]
        self._num_classes = max(self.labels) + 1

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_classes(self):
        return self._num_classes

    def len(self):
        return len(self.image_files)

    def get(self, idx):
        return self.converter.convert(self.image_files[idx], self.labels[idx])

# 3️⃣ **Efficient GNN Model**
class CaptchaGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GraphConv(num_features, 128)
        self.bn1 = BatchNorm(128)
        self.conv2 = GraphConv(128, 64)
        self.bn2 = BatchNorm(64)
        self.conv3 = GraphConv(64, 32)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.lin = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return F.log_softmax(self.lin(x), dim=-1)

# 4️⃣ **Training Function**
def train_model():
    dataset = CaptchaDataset("/content/drive/MyDrive/samples")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = CaptchaGNN(dataset.num_features, dataset.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    for epoch in range(50):
        model.train()
        train_loss = train_correct = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == data.y.squeeze()).sum().item()

        model.eval()
        test_correct = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                test_correct += (pred == data.y.squeeze()).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)
        print(f'Epoch {epoch:02d}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}')

        if test_acc >= 0.70:
            print(f"✅ Target reached: {test_acc:.2%} at epoch {epoch}")
            break

if __name__ == "__main__":
    train_model()
