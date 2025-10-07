# -------------------------
# Monte Carlo influence (efficient)
# -------------------------
import heapq
import pickle
from collections import defaultdict
from datetime import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
from typing import Tuple, Dict

from multiprocessing import Pool, cpu_count
from functools import partial
import heapq
from tqdm import tqdm

from monte_carlo import *
from ccar_calculate import monte_carlo_influence_and_ccar

from typing import List
import multiprocessing as mp

# We'll use a module-global variable for adj_T inside worker processes to avoid pickling large csr each task.
_ADJ_T = None

def _init_worker_for_rr(adj_T_data):
    """
    initializer for Pool: set global _ADJ_T from serialized components
    adj_T_data: tuple (data, indices, indptr, shape)
    """
    global _ADJ_T
    data, indices, indptr, shape = adj_T_data
    _ADJ_T = sp.csr_matrix((data, indices, indptr), shape=shape)

def _generate_one_rr(root_and_seed):
    """
    worker for generating one RR-set using global _ADJ_T.
    root_and_seed: (root, seed)
    returns list of nodes in RR-set
    """
    root, seed = root_and_seed
    rng = np.random.default_rng(seed)
    rr = set([root])
    stack = [root]
    adjT = _ADJ_T
    while stack:
        u = stack.pop()
        row_start, row_end = adjT.indptr[u], adjT.indptr[u+1]
        if row_end <= row_start:
            continue
        nbrs = adjT.indices[row_start:row_end]
        probs = adjT.data[row_start:row_end]
        # sample random numbers
        r = rng.random(len(nbrs))
        succ_mask = (r < probs)
        if not succ_mask.any():
            continue
        new_nodes = nbrs[succ_mask]
        for v in new_nodes:
            if v not in rr:
                rr.add(int(v))
                stack.append(int(v))
    return list(rr)


def generate_rr_sets(adj: sp.csr_matrix, num_rr: int = 20000, n_workers: int = None, rng_seed: int = 42):
    """
    Generate num_rr RR-sets for adj (IC model).
    Return: list_of_rr_sets (list of lists of node indices)
    Parallelized with multiprocessing Pool using initializer to avoid pickling adj repeatedly.
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    # transpose adjacency (reverse graph)
    adj_T = adj.transpose().tocsr()
    # serialize adj_T components for initializer
    adj_T_data = (adj_T.data, adj_T.indices, adj_T.indptr, adj_T.shape)

    # choose random roots and per-task seeds
    rng = np.random.default_rng(rng_seed)
    roots = rng.integers(low=0, high=adj_T.shape[0], size=num_rr)
    seeds = rng.integers(0, 2**31-1, size=num_rr)
    args = list(zip(roots.tolist(), seeds.tolist()))

    rr_sets = []
    if n_workers == 1:
        # serial fallback
        _init_worker_for_rr(adj_T_data)
        for arg in tqdm(args, desc="Generating RR-sets (serial)"):
            rr_sets.append(_generate_one_rr(arg))
    else:
        # parallel
        with Pool(processes=n_workers, initializer=_init_worker_for_rr, initargs=(adj_T_data,)) as pool:
            for rr in tqdm(pool.imap_unordered(_generate_one_rr, args), total=len(args), desc="Generating RR-sets"):
                rr_sets.append(rr)

    return rr_sets


def redistribute_leftover_by_degree(user_df,
                                    community_spent,
                                    deg,
                                    S,
                                    selected,
                                    leftover,
                                    adj,
                                    num_sim_refine,
                                    node2id=None,
                                    strict=True):
    """
    This function redistributes the remaining (leftover) budget after the main seed selection stage, named as LBA.
    Nodes are ranked by their degree adjusted by echo chamber ratio (ECR), and additional seeds are
    selected from unactivated nodes according to this score until the leftover budget is exhausted.
    It then evaluates the final influence spread and cross-community activation ratio (CCAR).
    """

    if 'node_idx' in user_df.columns:
        user_to_idx = dict(zip(user_df['UserIndex'].astype(str), user_df['node_idx'].astype(int)))
    elif node2id is not None:
        user_to_idx = dict(node2id)
    else:
        raise RuntimeError("redistribute_leftover_by_degree needs user_df obtain 'node_idx' or get node2id map")

    if deg is None:
        deg = np.diff(adj.indptr).astype(int)

    community_array = np.full(adj.shape[0], -1, dtype=int)
    for uid, c in zip(user_df['UserIndex'].astype(str), user_df['Community'].astype(int)):
        if uid in user_to_idx:
            idx_i = user_to_idx[uid]
            if 0 <= idx_i < adj.shape[0]:
                community_array[idx_i] = int(c)

    infl_res = monte_carlo_influence_and_ccar(adj, S, community_array=community_array, num_sim=1)
    activated_nodes = set(infl_res['activated_set'])  # monte_carlo_influence_and_ccar return the activated nodes

    candidates = []
    for _, row in user_df.iterrows():
        uid = str(row['UserIndex'])
        if uid not in user_to_idx:
            continue
        u_idx = int(user_to_idx[uid])
        if u_idx in S:
            continue

        if u_idx in activated_nodes:
            continue


        cost_u = float(row['User_cost'])
        d_u = int(deg[u_idx]) if u_idx < len(deg) else 0
        c = int(row['Community'])
        # candidates.append((d_u, cost_u, c, u_idx, uid))
        ecr = float(row['ECR']) if 'ECR' in row and not pd.isna(row['ECR']) else 0.0
        # print(ecr)
        score = d_u * (1.0 - 0.5 * ecr)
        candidates.append((score, cost_u, c, u_idx, uid, d_u, ecr))

    if not candidates:
        print("[Stage-2] no eligible candidate nodes (all nodes activated or selected).")
        community_array = np.full(adj.shape[0], -1, dtype=int)
        for uid, c in zip(user_df['UserIndex'].astype(str), user_df['Community'].astype(int)):
            if uid in user_to_idx:
                idx_i = user_to_idx[uid]
                if 0 <= idx_i < adj.shape[0]:
                    community_array[idx_i] = int(c)
        infl_res = monte_carlo_influence_and_ccar(adj, S, community_array=community_array, num_sim=num_sim_refine)
        print(f"[Stage-2] final influence = {int(round(infl_res['avg_activated']))}")
        print(f"[Stage-2] CCAR = {infl_res['avg_CCAR']:.4f}")
        return infl_res['avg_activated'], infl_res['avg_CCAR'], leftover, selected, S

    candidates.sort(key=lambda x: -x[0])

    total_spent = sum(community_spent.values())


    for score, cost_u, c, u_idx, uid, d_u, ecr in candidates:
        if leftover <= 1e-9:
            break
        if leftover >= cost_u:
            S.add(int(u_idx))
            selected.append(uid)
            community_spent[c] += cost_u
            total_spent += cost_u
            leftover -= cost_u
            used_any = True
            # print(f" -> Stage2 FULL assign node {uid} (idx {u_idx}, comm {c}, "
            #       f"deg={int(d_u)}, ECR={ecr:.3f}, score={score:.2f}, cost={cost_u:.2f}), "
            #       f"leftover={leftover:.4f}")
        else:
            if strict:
                continue
            else:
                S.add(int(u_idx))
                selected.append(uid)
                community_spent[c] += leftover
                total_spent += leftover
                print(f" -> Stage2 PARTIAL assign node {uid} (idx {u_idx}, comm {c}, "
                      f"deg={int(d_u)}, ECR={ecr:.3f}, score={score:.2f}, used={leftover:.4f}), "
                      f"leftover=0.0")
                leftover = 0.0
                used_any = True
                break

    print(f"[Stage-2 Done] total_spent (after stage2) = {total_spent:.4f}, leftover={leftover:.4f}")

    community_array = np.full(adj.shape[0], -1, dtype=int)
    for uid, c in zip(user_df['UserIndex'].astype(str), user_df['Community'].astype(int)):
        if uid in user_to_idx:
            idx_i = user_to_idx[uid]
            if 0 <= idx_i < adj.shape[0]:
                community_array[idx_i] = int(c)

    infl_res = monte_carlo_influence_and_ccar(adj, S, community_array=community_array, num_sim=num_sim_refine)

    print(f"\n[Final Influence after Stage-2] {int(round(infl_res['avg_activated']))} nodes influenced "
          f"(avg over {num_sim_refine} sims).")
    print(f"[Final CCAR after Stage-2] {infl_res['avg_CCAR']:.4f} "
          f"(cross-community edges ratio, avg over {num_sim_refine} sims).")

    return infl_res['avg_activated'], infl_res['avg_CCAR'], leftover, selected, S


def celf_rr_framework(adj: sp.csr_matrix,
                      user_df,
                      community_budgets: dict,
                      deg=None,
                      num_rr: int = 20000,
                      stage2_strict=True,
                      mc_runs=50,
                      num_sim_refine=200):
    """
    CELF using RR-set for Stage-1 initialization + Stage2 redistribution + MC statistics

    Supports two modes:
      - community-budget mode (original): community_budgets maps community_id -> budget
      - global-budget mode (baseline): pass community_budgets = {-1: B}
        then selection uses a single global budget (key -1).
    """
    start_time = time.time()
    n = adj.shape[0]
    node_cost = dict(zip(user_df['node_idx'], user_df['User_cost']))
    node_comm = dict(zip(user_df['node_idx'], user_df['Community']))

    # node_ecr = dict(zip(user_df['node_idx'], user_df.get('ECR', np.zeros(len(user_df)))))

    # Detect global mode: community_budgets has single special key -1
    global_key = -1
    global_mode = isinstance(community_budgets, dict) and (global_key in community_budgets)

    # -------- Stage-1: RR-based CELF --------
    rr_sets = generate_rr_sets(adj, num_rr=num_rr)
    m = len(rr_sets)

    node_to_rrs = defaultdict(list)
    for rid, rr in enumerate(rr_sets):
        for v in rr:
            node_to_rrs[v].append(rid)

    rr_covered = np.zeros(m, dtype=bool)
    cover_count = {u: len(node_to_rrs.get(u, [])) for u in node_to_rrs.keys()}

    # heap: (-score, node, last_updated_iter, last_marginal_count)
    heap = []
    for u, cnt in cover_count.items():
        cost_u = float(node_cost.get(u, 1.0))
        if cost_u <= 0:
            continue
        score = cnt / cost_u
        heapq.heappush(heap, (-score, u, 0, cnt))

    # community_spent used both in community mode and global mode (key -1)
    community_spent = defaultdict(float)
    S = set()
    selected = []
    selected_info = []
    iter_idx = 0

    def compute_marginal_uncovered(u):
        rr_list = node_to_rrs.get(u, [])
        cnt = sum(1 for rid in rr_list if not rr_covered[rid])
        return cnt

    # Main greedy selection loop
    while heap:
        neg_score, u, last_iter, last_cnt = heapq.heappop(heap)
        if u not in node_cost:
            continue
        cur_cnt = compute_marginal_uncovered(u)
        if cur_cnt != last_cnt:
            cost_u = float(node_cost.get(u, 1.0))
            heapq.heappush(heap, (-cur_cnt / cost_u, u, iter_idx, cur_cnt))
            continue

        # Determine budget key to check: community id or global_key
        comm = node_comm.get(u, None)
        if global_mode:
            # ignore community constraint, use global budget key
            comm_key = global_key
        else:
            comm_key = comm

        cost_u = float(node_cost.get(u, 1.0))

        # Check budget feasibility (works for both community and global modes)
        if comm_key is not None and community_spent[comm_key] + cost_u > community_budgets.get(comm_key, float('inf')):
            # cannot pick this node under current budget constraint -> skip
            continue

        if cur_cnt <= 0:
            break

        # select node
        S.add(u)
        selected.append(u)
        selected_info.append((u, cur_cnt, cost_u, comm if not global_mode else global_key))

        # update spent: if global_mode, update community_spent[-1] else update community_spent[comm]
        if comm_key is not None:
            community_spent[comm_key] += cost_u

        for rid in node_to_rrs.get(u, []):
            rr_covered[rid] = True
        iter_idx += 1

    # Stage-1 approximate influence
    final_infl_stage1 = (rr_covered.sum() / float(m)) * n
    print(f"[Stage-1] approx influence from RR={final_infl_stage1:.1f}")

    # Stage-2: redistribute leftover
    total_allocated = sum(community_budgets.values())
    total_spent = sum(community_spent.values())
    leftover = float(total_allocated - total_spent)

    final_infl, final_ccar, leftover_after, selected, S = redistribute_leftover_by_degree(
        user_df=user_df,
        community_spent=community_spent,
        deg=deg,
        S=S,
        selected=selected,
        leftover=leftover,
        adj=adj,
        num_sim_refine=num_sim_refine,
        strict=stage2_strict
    )

    # -------- Stage-2: skip redistribution (baseline without Stage-2) --------
    total_allocated = sum(community_budgets.values())
    total_spent = sum(community_spent.values())
    leftover = float(total_allocated - total_spent)

    print(f"[Stage-2 skipped] total_spent = {total_spent:.4f}, leftover = {leftover:.4f}")

    final_infl = final_infl_stage1

    run_time = time.time() - start_time


    # -------- Monte-Carlo statistics --------

    # DEBUG: check type and distribution of selected
    from collections import Counter
    types_cnt = Counter(type(x).__name__ for x in selected)
    print(f"[DEBUG] len(selected) = {len(selected)}, types = {types_cnt}, sample selected[:10] = {list(selected)[:10]}")

    userindex_to_nodeidx = {str(ui): int(nid) for ui, nid in zip(user_df['UserIndex'], user_df['node_idx'])}

    def normalize_selected_indices(selected_iter, userindex_to_nodeidx, adj_n):
        normalized = set()
        bad = []
        for u in selected_iter:
            if isinstance(u, (int, np.integer)):
                ui = int(u)
                if 0 <= ui < adj_n:
                    normalized.add(ui)
                else:
                    bad.append(u)
                continue

            if isinstance(u, str):

                try:
                    ui = int(u)
                    if 0 <= ui < adj_n:
                        normalized.add(ui)
                        continue
                except Exception:
                    pass

                if u in userindex_to_nodeidx:
                    normalized.add(userindex_to_nodeidx[u])
                    continue

                su = u.strip()
                if su in userindex_to_nodeidx:
                    normalized.add(userindex_to_nodeidx[su])
                    continue
                bad.append(u)
                continue

            try:
                ui = int(u)
                if 0 <= ui < adj_n:
                    normalized.add(ui)
                else:
                    bad.append(u)
            except Exception:
                bad.append(u)

        return normalized, bad

    adj_n = adj.shape[0]
    selected_norm, bad_items = normalize_selected_indices(selected, userindex_to_nodeidx, adj_n)

    if len(bad_items) > 0:
        print(
            f"[WARN] There are {len(bad_items)} selected items that couldn't be normalized to adj indices. sample: {bad_items[:10]}")

    print(f"[DEBUG] normalized selected size = {len(selected_norm)}, sample = {list(selected_norm)[:10]}")

    # Build community_array indexed by adj node id (0..n-1)
    community_array = np.zeros(adj_n, dtype=int)
    for _, row in user_df.iterrows():
        node_id = int(row['node_idx'])
        if 0 <= node_id < adj_n:
            community_array[node_id] = int(row['Community'])
        else:
            print(f"[WARN] user_df row has node_idx out of range: {node_id}")

    mc_results = []
    # rng = np.random.default_rng(42)
    for i in range(mc_runs):
        infl_res = monte_carlo_influence_and_ccar(adj, selected_norm,
                                                  community_array=community_array,
                                                  num_sim=1
                                                  )
        mc_results.append((infl_res['avg_activated'], infl_res['avg_CCAR'], infl_res['cross_edge_ratio'], infl_res['avg_Ecross']))
    mc_results = np.array(mc_results)
    mc_avg_infl = mc_results[:, 0].mean()
    mc_std_infl = mc_results[:, 0].std(ddof=1)
    mc_avg_ccar = mc_results[:, 1].mean()
    mc_std_ccar = mc_results[:, 1].std(ddof=1)
    mc_avg_cross_ratio = mc_results[:, 2].mean()
    mc_std_cross_ratio = mc_results[:, 2].std(ddof=1)
    mc_avg_Ecross = mc_results[:, 3].mean()
    mc_std_Ecross = mc_results[:, 3].std(ddof=1)

    print(f"CELF + Stage2 run time={run_time:.2f}s")
    print(
        f"[MC Final Results over {mc_runs} runs] "
        f"Influence={mc_avg_infl:.1f}±{mc_std_infl:.1f}, "
        f"CCAR={mc_avg_ccar:.4f}±{mc_std_ccar:.4f}, "
        f"CrossEdgeRatio={mc_avg_cross_ratio:.4f}±{mc_std_cross_ratio:.4f},"
        f"CrossEdges={mc_avg_Ecross:.1f}±{mc_std_Ecross:.1f}"
    )
    return selected, selected_info, float(final_infl), run_time, mc_avg_infl, mc_std_infl, mc_avg_ccar, mc_std_ccar, mc_avg_cross_ratio, mc_std_cross_ratio, mc_avg_Ecross, mc_std_Ecross



