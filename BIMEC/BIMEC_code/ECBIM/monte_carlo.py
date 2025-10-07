import numpy as np
import scipy.sparse as sp

def compute_node_gain(u, idx_to_cost, current_infl, adj, num_sim):
    cost_u = idx_to_cost[u]
    if cost_u <= 0:
        return None
    gain = monte_carlo_influence(adj, {u}, num_sim=num_sim)
    marginal = gain - current_infl
    ratio = marginal / cost_u
    return (-ratio, u, marginal, 0)  # CELF heap

def compute_gain_for_node(u, S, adj, idx_to_cost, num_sim):
    cost_u = idx_to_cost[u]
    infl_with = monte_carlo_influence(adj, S | {u}, num_sim=num_sim)
    infl_without = monte_carlo_influence(adj, S, num_sim=num_sim)
    gain = infl_with - infl_without
    ratio = gain / cost_u if cost_u > 0 else 0.0
    return (-ratio, u, gain, len(S))

def monte_carlo_influence(adj: sp.csr_matrix, seeds: set, num_sim: int = 100, rng=None):
    """
    Monte Carlo simulation for IC model.
    return the mean activated nodes（float）
    """
    if rng is None:
        rng = np.random.default_rng()

    total_activated = 0.0
    getrow = adj.getrow

    for _ in range(num_sim):
        activated = set(seeds)
        frontier = list(seeds)
        while frontier:
            next_frontier = []
            for u in frontier:
                row = getrow(u)
                neighbors = row.indices
                probs = row.data
                if neighbors.size == 0:
                    continue
                r = rng.random(neighbors.size)
                succ_mask = (r < probs)
                if not succ_mask.any():
                    continue
                new_nodes = neighbors[succ_mask]
                for v in new_nodes:
                    if v not in activated:
                        activated.add(int(v))
                        next_frontier.append(int(v))
            frontier = next_frontier
        total_activated += len(activated)

    return total_activated / float(num_sim)

# ---------- return activated set ----------
def simulate_activation_once(adj: sp.csr_matrix, seeds: set, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    activated = set(int(u) for u in seeds)
    frontier = list(seeds)
    getrow = adj.getrow

    while frontier:
        next_frontier = []
        for u in frontier:
            row = getrow(int(u))
            neighbors = row.indices
            probs = row.data
            if neighbors.size == 0:
                continue
            r = rng.random(neighbors.size)
            succ_mask = (r < probs)
            if not succ_mask.any():
                continue
            new_nodes = neighbors[succ_mask]
            for v in new_nodes:
                vi = int(v)
                if vi not in activated:
                    activated.add(vi)
                    next_frontier.append(vi)
        frontier = next_frontier
    return activated
