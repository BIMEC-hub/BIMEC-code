import math
from data_loader import load_user_data
from data_loader import load_edge_prob
from budget_allocation import allocate_community_budgets
from influence_maximization import celf_rr_framework
# from influence_maximization import celf_greedy
import random, numpy as np, torch



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

    # user_file = 'UserCost_Calculate/Weibo/user_cost_weibo_15.txt'
    # edge_file = '../EdgeProcess/Edge_mean_0_01/edge_weibo_scaled_p0_01_our.txt'

    user_file = 'UserCost_Calculate/Twitter15/user_cost_Twitter15_15.txt'
    edge_file = '../EdgeProcess/Edge_mean_0_01/edge_twitter15_scaled_p0_01_our.txt'


    user_df  = load_user_data(user_file)  ############# normal

    # user_df = load_user_data(user_file, fixed_cost=1.0) #########


    user_df = user_df.reset_index(drop=True).copy()
    node2id = {u: i for i, u in enumerate(user_df["UserIndex"])}
    user_df["node_idx"] = user_df["UserIndex"].map(node2id)

    total_cost = user_df['User_cost'].sum()
    print(f"original total cost = {total_cost}")

    # baseline random cost ======================
    # rng = np.random.default_rng(seed=42)
    # baseline_random_costs = rng.uniform(1, 20, size=len(user_df))
    #
    # # rescale to the total budget
    # scale = total_cost / baseline_random_costs.sum()
    # baseline_random_costs *= scale
    #
    # user_df['User_cost'] = baseline_random_costs
    #=======================================================
    # user_df['User_cost'] = total_cost / len(user_df)
    # user_df['User_cost'] = np.random.randint(1, 20, size=len(user_df))

    # =======================================================

    adj = load_edge_prob(edge_file, user_df, node2id, fixed_prob=None)

    budget_fractions = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05] # total budget ratio (adjustable)
    # budget_fractions = [0.001] # total budget ratio (adjustable)
    # budget_fractions = [0.2] # total budget ratio (adjustable)

    # Monte Carlo index (adjustable)
    num_sim_init = 40  # CELF Initial fast estimate times (low)
    num_sim_refine = 200  # CELF Final Evaluation Times (high)
    tol = 1e-3

    total_cost_sum = float(user_df['User_cost'].sum())
    print(f"Total user cost sum = {total_cost_sum:.4f}")

    # run every budget assignment ratio once + CELF
    for frac in budget_fractions:
        B = math.floor(total_cost_sum * frac)
        print("\n")
        print(f"Running for total budget fraction = {frac}  \n Total budget B = {B}")
        # community_budgets = {-1: B}
        community_budgets, H_c, hatH, deg = allocate_community_budgets(user_df, adj, total_budget=B, alpha=1.5, gamma=1)

        # print every allocated budgets for each community
        # print("Community budgets (community : budget) :")
        # for c, b in sorted(community_budgets.items()):
        #     print(f"  community {c} : budget = {b:.2f}")

        # run CELF greedy with MC repeat evaluation
        selected, info, final_infl, run_time, mc_avg_infl, mc_std_infl, mc_avg_ccar, mc_std_ccar, mc_avg_cross_ratio, mc_std_cross_ratio, mc_avg_Ecross, mc_std_Ecross = celf_rr_framework(
            adj,
            user_df,
            community_budgets,
            # num_sim_init=40,
            num_sim_refine=500,
            # tol=1e-3,
            # time_limit=None,
            deg=deg,
            # deg=None,
            stage2_strict=False,
            mc_runs=500
        )

        # output the result
        print("\n--- RESULTS ---")
        print(f"budget ratio = {frac}, total budget B = {B}")
        # print("Selected seeds:", selected)
        print(f"Stage-2 single-run activated: {math.floor(final_infl)}")
        print(f"Monte Carlo average activated: {mc_avg_infl:.1f} ± {mc_std_infl:.1f}")
        print(f"Monte Carlo average CCAR: {mc_avg_ccar:.4f} ± {mc_std_ccar:.4f}")
        print(f"Monte Carlo average cross_ratio: {mc_avg_cross_ratio:.4f} ± {mc_std_cross_ratio:.4f}")
        print(f"Monte Carlo average cross edges: {mc_avg_Ecross:.4f} ± {mc_std_Ecross:.4f}")
        print(f"Whole running time: {run_time:.2f} seconds")
