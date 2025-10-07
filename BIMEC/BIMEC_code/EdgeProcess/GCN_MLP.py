import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from load_data import load_graph_data

class EdgePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.gcn1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.gcn2(h, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)

        src, dst = edge_index
        edge_feat = torch.cat([h[src], h[dst]], dim=1)
        edge_prob = self.mlp(edge_feat).squeeze()
        return edge_prob, h

# ===================== Margin-based edge loss =====================
def margin_edge_loss(edge_prob, edge_index, community_ids, margin=0.1):
    src, dst = edge_index
    same_mask = (community_ids[src] == community_ids[dst])
    cross_mask = ~same_mask

    edge_intra = edge_prob[same_mask]
    edge_inter = edge_prob[cross_mask]

    if len(edge_intra) == 0 or len(edge_inter) == 0:
        return torch.tensor(0.0, device=edge_prob.device)

    loss = F.relu(margin - edge_intra.mean() + edge_inter.mean())
    return loss

def train_and_save(edge_file, community_file, save_file,
                   hidden_dim=64, epochs=500, lr=1e-3, margin=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data, edge_index, node2comm, edges_df, node2idx = load_graph_data(edge_file, community_file)
    data = data.to(device)
    # community_ids corresponding to the order of all_nodes
    all_nodes = sorted(node2idx.keys())
    community_ids = torch.tensor([node2comm[node] for node in all_nodes], dtype=torch.long, device=device)

    model = EdgePredictor(in_channels=data.num_features, hidden_channels=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        edge_prob, h = model(data)
        loss = margin_edge_loss(edge_prob, data.edge_index, community_ids, margin=margin)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                src, dst = data.edge_index
                same_mask = (community_ids[src] == community_ids[dst])
                inter_mask = ~same_mask
                intra_mean = edge_prob[same_mask].mean().item()
                inter_mean = edge_prob[inter_mask].mean().item()
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | intra: {intra_mean:.4f} | inter: {inter_mean:.4f}")

    model.eval()
    with torch.no_grad():
        edge_prob, _ = model(data)
    edges_df["pred_prob"] = edge_prob.cpu().numpy()
    edges_df.to_csv(save_file, sep="\t", index=False)
    print("Predicted edge probabilities saved to", save_file)


if __name__ == "__main__":
    # edge_file = '../Datasets/AskUbuntu/ubuntu_graph_filtered.txt'
    # community_file = '../Datasets/AskUbuntu/communities_ubuntu_louvain_filtered.txt'
    # save_file = 'EdgeResult/edge_ubuntu.txt'

    # edge_file = '../Datasets/Weibo/topo.txt'
    # community_file = '../Datasets/Weibo/communities_weibo.txt'
    # save_file = 'EdgeResult/edge_weibo.txt'

    # edge_file = '../Datasets/Twitter-19/twitter_covid19_graph_filtered.txt'
    # community_file = '../Datasets/Twitter-19/communities_covid19_louvain_filtered.txt'
    # save_file = 'EdgeResult/edge_twitter19.txt'

    edge_file = '../Datasets/YouTube/youtube_top5000_subgraph.txt'
    community_file = '../Datasets/YouTube/node2comm_youtube.txt'
    save_file = 'EdgeResult/edge_youtube.txt'

    train_and_save(edge_file, community_file, save_file)






