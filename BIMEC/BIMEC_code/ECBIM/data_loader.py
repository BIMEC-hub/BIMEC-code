from io import StringIO
import pandas as pd
import scipy.sparse as sp
import numpy as np

import pandas as pd
from io import StringIO


def load_user_data(user_file, fixed_cost=None):
    """
    Read echo_chamber_results.txt (including the header line) and return a DataFrame containing ['UserIndex', 'Community', 'User_cost'].
    Use sep=r'\s+' to support both space and tab delimiters.

    Parameters:
        fixed_cost (float or None): If set to a numeric value, all User_cost entries will be assigned this fixed value;
        if None, the original values from the file will be used.
    """
    with open(user_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("UserIndex"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"can't find 'UserIndex' header in {user_file}")

    data = "".join(lines[header_idx:])
    df = pd.read_csv(StringIO(data), sep=r"\s+", engine="python")

    expected_cols = ['UserIndex', 'Community', 'User_cost', 'ECR']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing row: {missing}, real row: {df.columns.tolist()}")

    df = df[expected_cols].copy()
    df['UserIndex'] = df['UserIndex'].astype(str)
    df['Community'] = df['Community'].astype(int)
    df['User_cost'] = df['User_cost'].astype(float)
    df['ECR'] = df['ECR'].astype(float)

    # if there is fix value, cover the User_cost
    if fixed_cost is not None:
        df['User_cost'] = fixed_cost

    return df

def load_edge_prob(edge_file, user_df, node2id, fixed_prob=None):
    """
    Read the edge file (node1, node2, probability, softmax_weight).
    If fixed_prob is not None, ignore the probability values in the file and assign all edges with fixed_prob.
    Return a csr_matrix (num_nodes × num_nodes) whose values represent the directional probabilities.
    """
    df = pd.read_csv(edge_file, sep=r"\s+", engine="python", header=0)

    if 'node1' not in df.columns or 'node2' not in df.columns:
        raise RuntimeError("edge file need col: node1 node2 probability")

    df['node1'] = df['node1'].astype(str).map(node2id)
    df['node2'] = df['node2'].astype(str).map(node2id)

    if df['node1'].isnull().any() or df['node2'].isnull().any():
        missing_nodes = set(df.loc[df['node1'].isnull(), 'node1']).union(
                        set(df.loc[df['node2'].isnull(), 'node2']))
        raise ValueError(f"Edge file has unknown nodes: {missing_nodes}")

    row = df['node1'].astype(int).values
    col = df['node2'].astype(int).values

    if fixed_prob is not None:
        data = np.full(len(row), float(fixed_prob), dtype=float)
    else:
        if 'pred_prob' not in df.columns:
            raise RuntimeError("edge file need to contain pred_prob col or set a fixed_prob")
        data = df['pred_prob'].astype(float).values

    num_nodes = len(user_df)
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj
