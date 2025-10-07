import torch
import pandas as pd
from torch_geometric.data import Data


def load_graph_data(edge_file, community_file):
    edges_df = pd.read_csv(edge_file, sep=r"\s+", engine="python")
    edges_df["node1"] = edges_df["node1"].astype(str)
    edges_df["node2"] = edges_df["node2"].astype(str)

    communities_df = pd.read_csv(community_file, sep=r"\s+", header=None,
                                 names=["node", "community"], engine="python")
    communities_df["node"] = communities_df["node"].astype(str)

    all_nodes = sorted(set(edges_df["node1"]).union(set(edges_df["node2"])))
    node2idx = {node: i for i, node in enumerate(all_nodes)}
    node2comm = dict(zip(communities_df["node"], communities_df["community"]))

    edge_index = torch.tensor(
        [[node2idx[u], node2idx[v]] for u, v in zip(edges_df["node1"], edges_df["node2"])],
        dtype=torch.long
    ).t().contiguous()

    num_nodes = len(all_nodes)
    degrees = torch.zeros((num_nodes, 1))
    for u, v in edge_index.t():
        degrees[u] += 1
        degrees[v] += 1
    degrees_scaled = degrees / degrees.max()  # rescale to [0,1]
    # community embedding
    num_comms = max(node2comm.values()) + 1
    community_emb = torch.nn.functional.one_hot(
        torch.tensor([node2comm[node] for node in all_nodes]), num_classes=num_comms
    ).float()

    x = torch.cat([degrees_scaled, community_emb], dim=1)
    data = Data(x=x, edge_index=edge_index)

    return data, edge_index, node2comm, edges_df, node2idx