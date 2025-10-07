import math
from data_loader import load_user_data
from data_loader import load_edge_prob
from budget_allocation import allocate_community_budgets
from influence_maximization import celf_rr_framework
import random, numpy as np, torch
# from baseline_globalLBA import lba_global_baseline

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # user_file = 'UserCost_Calculate/Twitter19/user_cost_Twitter19_15.txt'
    # edge_file = '../EdgeProcess/Edge_mean_0_01/edge_twitter19_scaled_p0_01_our.txt'

    # user_file = 'UserCost_Calculate/Ubuntu/user_cost_AskUbuntu_15.txt'
    # edge_file = '../EdgeProcess/Edge_mean_0_01/edge_ubuntu_scaled_p0_01_our.txt'

    # user_file = 'UserCost_Calculate/Youtube/user_cost_Youtube_15.txt'
    # edge_file = '../EdgeProcess/Edge_mean_0_01/edge_youtube_scaled_p0_01_our.txt'

    user_file = 'UserCost_Calculate/Twitter15/user_cost_Twitter15_15.txt'
    edge_file = '../EdgeProcess/Edge_mean_0_01/edge_twitter15_scaled_p0_01_our.txt'

    user_df  = load_user_data(user_file)  # normal
    user_df = user_df.reset_index(drop=True).copy()
    node2id = {u: i for i, u in enumerate(user_df["UserIndex"])}
    user_df["node_idx"] = user_df["UserIndex"].map(node2id)

    total_cost = user_df['User_cost'].sum()
    print(f"Original total cost = {total_cost}")

    adj = load_edge_prob(edge_file, user_df, node2id, fixed_prob=None)

    budget_fractions = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

    num_sim_refine = 500
    mc_runs = 500

    total_cost_sum = float(user_df['User_cost'].sum())
    print(f"Total user cost sum = {total_cost_sum:.4f}")

    for frac in budget_fractions:
        B = math.floor(total_cost_sum * frac)
        print("\n")
        print(f"Running for total budget fraction = {frac}  \n Total budget B = {B}")

        # budget allocation
        community_budgets, H_c, hatH, deg = allocate_community_budgets(
            user_df, adj, total_budget=B, alpha=1.5, gamma=1
        )

        # # ---------------- BIMEC ORIGINAL CELF-RR ----------------
        # selected, info, final_infl, run_time, mc_avg_infl, mc_std_infl, \
        # mc_avg_ccar, mc_std_ccar, mc_avg_cross_ratio, mc_std_cross_ratio, \
        # mc_avg_Ecross, mc_std_Ecross = celf_rr_framework(
        #     adj,
        #     user_df,
        #     community_budgets,
        #     num_sim_refine=num_sim_refine,
        #     deg=deg,
        #     stage2_strict=False,
        #     mc_runs=mc_runs
        # )
        #

        # ---------------- big element ----------------
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        print("\n--- BIG NODE CONTRIBUTION ANALYSIS ---")
        for alpha in alpha_list:
            big_nodes = []
            for c, b_i in community_budgets.items():
                nodes_in_c = user_df[user_df['Community'] == c]
                big_nodes_c = nodes_in_c[nodes_in_c['User_cost'] >= alpha * b_i]['UserIndex'].tolist()
                big_nodes.extend(big_nodes_c)
            print(f"Alpha = {alpha}, number of big nodes = {len(big_nodes)}")

            if len(big_nodes) == 0:
                print("No big nodes for this alpha.")
                continue

            selected, info, final_infl, run_time, mc_avg_infl, mc_std_infl, \
            mc_avg_ccar, mc_std_ccar, mc_avg_cross_ratio, mc_std_cross_ratio, \
            mc_avg_Ecross, mc_std_Ecross = celf_rr_framework(
                adj,
                user_df[user_df['UserIndex'].isin(big_nodes)],
                community_budgets,
                num_sim_refine=num_sim_refine,
                deg=deg,
                stage2_strict=False,
                mc_runs=mc_runs
            )
            print(f"Delta_big (influence of big nodes) = {mc_avg_infl:.1f}")
