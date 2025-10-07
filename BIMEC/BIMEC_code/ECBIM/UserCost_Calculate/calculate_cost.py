import pandas as pd
from collections import defaultdict
import os


def compute_user_cost(user_file, edge_file, lambda_=1.0, out_file="user_cost.txt"):

    user_df = pd.read_csv(user_file, sep=r"\s+", header=None, names=["UserIndex", "Community"], dtype=str)
    user_df["Community"] = user_df["Community"].astype(int)

    # user -> community mapping
    user2comm = dict(zip(user_df["UserIndex"], user_df["Community"]))

    edges = pd.read_csv(edge_file, sep=r"\s+", header=None, names=["node1", "node2"], dtype=str)

    # Build neighbor list
    neighbors = defaultdict(set)
    for u, v in edges.itertuples(index=False):
        neighbors[u].add(v)
        neighbors[v].add(u)  # undirected graph

    # Compute Echo Chamber Ratio (ECR)
    ecr_list = []
    for u in user_df["UserIndex"]:
        neighs = neighbors.get(u, [])
        if not neighs:
            ecr = 0.0  # no neighbors
        else:
            # Count neighbor community distribution
            comm_counts = defaultdict(int)
            for v in neighs:
                if v in user2comm:  # only count users present in user list
                    comm_counts[user2comm[v]] += 1

            total = sum(comm_counts.values())
            if total == 0:
                ecr = 0.0
            else:
                proportions = {c: cnt / total for c, cnt in comm_counts.items()}
                own_c = user2comm[u]
                p_prime = proportions.get(own_c, 0.0)

                if len(proportions) == 1:
                    ecr = 1.0  # all neighbors belong to the same community
                else:
                    other_ps = [p for c, p in proportions.items() if c != own_c]
                    ecr = sum(abs(p_prime - pi) for pi in other_ps) / (len(proportions) - 1)
                    ecr = ecr * p_prime

        ecr_list.append(ecr)

    # Compute final user cost
    user_df["ECR"] = ecr_list
    user_df["User_cost"] = 1 + lambda_ * user_df["ECR"]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    user_df.to_csv(out_file, sep="\t", index=False, columns=["UserIndex", "Community", "ECR", "User_cost"])
    print(f"Saved to {out_file}")


# --------------------------
# Batch processing section
# --------------------------

datasets = {
    "Twitter15": {
        "user_file": "../../Datasets/Twitter-15/communities_louvain_filtered.txt",
        "edge_file": "../../Datasets/Twitter-15/twitter_10_graph_filtered.txt",
        "lambdas": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0],
    },
    # "Twitter19": {
    #     "user_file": "../../Datasets/COVID-19/communities_covid19_louvain_filtered.txt",
    #     "edge_file": "../../Datasets/COVID-19/twitter_covid19_graph_filtered.txt",
    #     "lambdas": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
    # },
    # "Youtube": {
    #     "user_file": "../../Datasets/YouTube/node2comm_youtube.txt",
    #     "edge_file": "../../Datasets/YouTube/youtube_top5000_subgraph.txt",
    #     "lambdas": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
    # },
     #   "AskUbuntu": {
    #     "user_file": "../../Datasets/AskUbuntu/communities_ubuntu_louvain_filtered.txt",
    #     "edge_file": "../../Datasets/AskUbuntu/ubuntu_graph_filtered.txt",
    #     "lambdas": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
    # },
}

for name, info in datasets.items():
    for lam in info["lambdas"]:
        out_dir = f"{name}/user_cost_{name}_{lam}.txt"
        compute_user_cost(info["user_file"], info["edge_file"], lambda_=lam, out_file=out_dir)
