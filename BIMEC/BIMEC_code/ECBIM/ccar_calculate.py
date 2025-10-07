import numpy as np
import scipy.sparse as sp
from typing import Dict, Set

def monte_carlo_influence_and_ccar(adj: sp.csr_matrix,
                                        seeds: Set[int],
                                        community_array: np.ndarray,
                                        num_sim: int = 100,
                                        rng=None) -> Dict[str, object]:
    """
    Optimized Monte Carlo simulation for IC model with CCAR measurement.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = adj.shape[0]
    indptr, indices, data = adj.indptr, adj.indices, adj.data

    activations = np.zeros(num_sim, dtype=float)
    Eacts = np.zeros(num_sim, dtype=float)
    Ecrosses = np.zeros(num_sim, dtype=float)

    total_edges = adj.nnz

    for s in range(num_sim):
        local_rng = rng
        activated = np.zeros(n, dtype=bool)
        activated[list(seeds)] = True
        frontier = np.array(list(seeds), dtype=np.int32)

        Eact_count = 0
        Ecross_count = 0

        while frontier.size > 0:
            next_frontier = []

            for u in frontier:
                start, end = indptr[u], indptr[u + 1]
                neighbors = indices[start:end]
                probs = data[start:end]

                if neighbors.size == 0:
                    continue

                # sample activation
                succ_mask = (local_rng.random(neighbors.size) < probs)
                if not np.any(succ_mask):
                    continue

                new_nodes = neighbors[succ_mask]
                new_nodes = new_nodes[~activated[new_nodes]]
                if new_nodes.size == 0:
                    continue

                activated[new_nodes] = True
                next_frontier.extend(new_nodes.tolist())

                # update counters
                Eact_count += new_nodes.size
                cu = community_array[u]
                if cu != -1:
                    cvs = community_array[new_nodes]
                    Ecross_count += np.count_nonzero((cvs != -1) & (cvs != cu))

            frontier = np.array(next_frontier, dtype=np.int32)

        activations[s] = np.sum(activated)
        Eacts[s] = Eact_count
        Ecrosses[s] = Ecross_count

        if s == num_sim - 1:
            final_activated_set = np.where(activated)[0].tolist()

    # per-sim CCAR
    ccars = np.zeros(num_sim, dtype=float)
    nonzero_mask = (Eacts > 0)
    ccars[nonzero_mask] = Ecrosses[nonzero_mask] / Eacts[nonzero_mask]

    cross_edge_ratio = np.mean(Ecrosses) / total_edges

    return {
        'avg_activated': float(np.mean(activations)),
        'avg_Eact': float(np.mean(Eacts)),
        'avg_Ecross': float(np.mean(Ecrosses)),
        'avg_CCAR': float(np.mean(ccars)),
        'cross_edge_ratio': float(cross_edge_ratio),
        'activations': activations,
        'Eacts': Eacts,
        'Ecrosses': Ecrosses,
        'ccars': ccars,
        'activated_set': final_activated_set
    }
