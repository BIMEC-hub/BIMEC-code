import math
from collections import defaultdict
import numpy as np


def allocate_community_budgets(user_df, adj, total_budget, alpha=1.0, gamma=1.0):
    """
    Allocate community budgets based on NCE (with community size weighting),
    preserving floating-point precision and evenly distributing any remaining budget.

    Parameters:
        alpha: Exponent for community entropy weighting
        gamma: Exponent for community size weighting
    """
    n = adj.shape[0]
    comm_arr = -np.ones(n, dtype=int)
    for _, row in user_df.iterrows():
        comm_arr[int(row['node_idx'])] = int(row['Community'])
    communities = sorted(user_df['Community'].unique())
    comm_to_idx = {c: i for i, c in enumerate(communities)}
    K = len(communities)
    logK = math.log(K) if K > 1 else 1.0

    # compute d_u and hatH(u)
    hatH = np.zeros(n, dtype=float)
    deg = np.zeros(n, dtype=float)
    getrow = adj.getrow
    for u in range(n):
        row = getrow(u)
        neighbors = row.indices
        deg[u] = neighbors.size
        if neighbors.size == 0:
            hatH[u] = 0.0
            continue
        counts = np.zeros(K, dtype=float)
        for v in neighbors:
            c = int(comm_arr[int(v)])
            if c == -1:
                continue
            counts[comm_to_idx[c]] += 1.0
        d_u = counts.sum()
        if d_u == 0:
            hatH[u] = 0.0
            continue
        p = counts / d_u
        mask = p > 0
        H = -np.sum(p[mask] * np.log(p[mask]))
        hatH[u] = H / logK

    # compute community entropy H_c (degree-weighted) and degree-sum S_c
    sum_d_hatH = defaultdict(float)
    sum_d = defaultdict(float)
    sum_cost_by_comm = defaultdict(float)
    for _, row in user_df.iterrows():
        u = int(row['node_idx'])
        c = int(row['Community'])
        sum_d_hatH[c] += deg[u] * hatH[u]
        sum_d[c] += deg[u]
        sum_cost_by_comm[c] += float(row['User_cost'])

    H_c = {}
    S_c = {}
    for c in communities:
        H_c[c] = (sum_d_hatH[c] / sum_d[c]) if sum_d[c] > 0 else 0.0
        S_c[c] = sum_d[c]

    # compute hybrid weights w_c = H_c^alpha * S_c^gamma
    w_c = {c: (H_c[c]**alpha) * (S_c[c]**gamma) for c in communities}
    sum_w = sum(w_c.values())

    community_budgets = {}
    if sum_w <= 0:
        total_cost = user_df['User_cost'].sum()
        for c in communities:
            alloc_raw = total_budget * (sum_cost_by_comm[c] / (total_cost + 1e-12))
            community_budgets[c] = min(sum_cost_by_comm[c], alloc_raw)
    else:
        for c in communities:
            alloc_raw = total_budget * (w_c[c] / sum_w)
            community_budgets[c] = min(sum_cost_by_comm[c], alloc_raw)

    return community_budgets, H_c, hatH, deg
