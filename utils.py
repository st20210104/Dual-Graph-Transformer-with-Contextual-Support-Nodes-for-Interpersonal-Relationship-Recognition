import torch
import dgl
import random
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from feature_extraction import FeatureExtractor

extractor = FeatureExtractor(device='cpu')

def create_pair_centric_graph(personA_img=None, personB_img=None, num_features=128):
    g = dgl.graph(([], []))
    g.add_nodes(3)
    if personA_img and personB_img:
        pair_feat = extractor.extract_pair_centric_features(personA_img, personB_img)
        reduced_feat = torch.randn(3, num_features)  # Replace with projection
        g.ndata['feat'] = reduced_feat
    else:
        g.ndata['feat'] = torch.randn(3, num_features)
    return g

def create_scene_centric_graph(scene_img=None, num_features=128):
    g = dgl.graph(([], []))
    g.add_nodes(4)
    if scene_img:
        visual_feat, text_feat = extractor.extract_scene_centric_features(scene_img)
        reduced_feat = torch.randn(4, num_features)  # Replace with projection
        g.ndata['feat'] = reduced_feat
    else:
        g.ndata['feat'] = torch.randn(4, num_features)
    return g

class IPRDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, num_features=128, num_classes=6, image_paths=None, labels=None):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.image_paths and self.labels:
            personA_img, personB_img, scene_img = self.image_paths[idx]
            label = self.labels[idx]
            pair_graph = create_pair_centric_graph(personA_img, personB_img, self.num_features)
            scene_graph = create_scene_centric_graph(scene_img, self.num_features)
        else:
            pair_graph = create_pair_centric_graph(num_features=self.num_features)
            scene_graph = create_scene_centric_graph(num_features=self.num_features)
            label = random.randint(0, self.num_classes - 1)
        return pair_graph, scene_graph, label

def collate_fn(batch):
    pair_graphs, scene_graphs, labels = zip(*batch)
    bg_pair = dgl.batch(pair_graphs)
    bg_scene = dgl.batch(scene_graphs)
    labels = torch.tensor(labels)
    return bg_pair, bg_scene, labels

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for pair_graphs, scene_graphs, labels in dataloader:
            pair_graphs = pair_graphs.to(device)
            scene_graphs = scene_graphs.to(device)
            labels = labels.to(device)
            outputs = model(pair_graphs, scene_graphs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    per_class_acc = {cls: accuracy_score(
        [l for i, l in enumerate(all_labels) if l == cls],
        [p for i, p in enumerate(all_preds) if all_labels[i] == cls]
    ) for cls in set(all_labels)}
    y_true_bin = np.eye(num_classes)[all_labels]
    mAP = average_precision_score(y_true_bin, np.array(all_probs), average='macro')
    return acc, per_class_acc, mAP
