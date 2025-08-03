import torch
import torch.nn as nn
import dgl
from dgl.nn import GatedGraphConv

class DualGraphTransformerIPR(nn.Module):
    def __init__(self, in_feats, hidden_size, n_steps, n_classes):
        super(DualGraphTransformerIPR, self).__init__()
        self.ggnn_pair = GatedGraphConv(in_feats, hidden_size, n_steps, n_etypes=1)
        self.ggnn_scene = GatedGraphConv(in_feats, hidden_size, n_steps, n_etypes=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, pair_graph, scene_graph):
        h_pair = self.ggnn_pair(pair_graph, pair_graph.ndata['feat'],
                                torch.zeros(pair_graph.number_of_edges(), dtype=torch.long))
        h_scene = self.ggnn_scene(scene_graph, scene_graph.ndata['feat'],
                                  torch.zeros(scene_graph.number_of_edges(), dtype=torch.long))
        pair_graph.ndata['h'] = h_pair
        scene_graph.ndata['h'] = h_scene
        pair_emb = dgl.mean_nodes(pair_graph, 'h')
        scene_emb = dgl.mean_nodes(scene_graph, 'h')
        fusion_input = torch.stack([pair_emb, scene_emb], dim=0)
        fusion_out = self.transformer(fusion_input)
        fusion_vec = fusion_out.mean(dim=0)
        return self.fc(fusion_vec)
