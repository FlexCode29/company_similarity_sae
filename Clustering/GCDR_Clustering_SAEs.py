import plotly.graph_objects as go
import kaleido
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd
import psutil
import os
import joblib
import optuna
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import json
from sklearn.preprocessing import StandardScaler
import json
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
from functools import partial
from joblib import Parallel, delayed
import logging
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from functools import partial
from joblib import Parallel, delayed
import logging
import os


import warnings
warnings.filterwarnings("ignore")
import psutil
import os
process = psutil.Process(os.getpid())
print(f"Memory usage before: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Helper functions:
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    filename='optuna_optimization.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def calculate_avg_correlation(TO_ANALYSE_DF, cluster_df, cluster_type):
    """
    Calculate the mean correlation for each cluster type per year.

    Parameters:
    - TO_ANALYSE_DF (pd.DataFrame): Original DataFrame containing 'year', 'Company1', 'Company2', and 'correlation'.
    - cluster_df (pd.DataFrame): DataFrame with 'year' and 'clusters' columns.
    - cluster_type (str): Description of the cluster type for labeling.

    Returns:
    - pd.DataFrame: DataFrame with 'year' and average correlation for the cluster type.
    """
    avg_correlations = []

    for _, row in tqdm(cluster_df.iterrows(), desc=f"Calculating Stats for {cluster_type}", total=len(cluster_df)):
        year = row['year']
        clusters = row['clusters']
        year_data = TO_ANALYSE_DF[TO_ANALYSE_DF['year'] == year]
        cluster_stats = []

        for cluster_id, companies in clusters.items():
            if len(companies) <= 1:  # Skip clusters with only 1 company
                continue

            # Get all pairs of companies within the cluster
            cluster_pairs = year_data[
                (year_data['Company1'].isin(companies) & year_data['Company2'].isin(companies))
            ]
            # Calculate statistics for the cluster
            if not cluster_pairs.empty:
                try:
                    correlations = cluster_pairs['correlation']
                except KeyError:
                    # Handle alternative column name if 'correlation' doesn't exist
                    correlations = cluster_pairs['ActualCorrelation']
                cluster_stats.append(correlations.mean())

        # Aggregate statistics across clusters for the year
        if cluster_stats:
            avg_correlations.append({
                'year': year,
                f'{cluster_type}AvgCorrelation': sum(cluster_stats) / len(cluster_stats)
            })
        else:
            avg_correlations.append({
                'year': year,
                f'{cluster_type}AvgCorrelation': np.nan
            })

    return pd.DataFrame(avg_correlations)


def perform_clustering_per_year(
    TO_ANALYSE_DF,
    years_to_cluster,
    threshold,
    linkage_method='single'
):
    """
    Perform clustering on specified years using MST thresholding.

    Parameters:
    - pairs_df (pd.DataFrame): DataFrame containing company pairs with 'Company1', 'Company2', 'year', and 'sum_abs_diff_scaled_01'.
    - years_to_cluster (list): List of years to perform clustering on.
    - threshold (float): Threshold for forming clusters by removing edges from the MST.
    - linkage_method (str): Linkage method for hierarchical clustering ('single', 'complete', 'average', 'ward').

    Returns:
    - pd.DataFrame: DataFrame with 'year' and 'clusters' columns.
    """

    # Filter the DataFrame for the specified years
    filtered_df = TO_ANALYSE_DF[TO_ANALYSE_DF['year'].isin(years_to_cluster)]

    # Get sorted list of unique years within the specified subset
    unique_years = sorted(filtered_df['year'].unique())

    # Initialize list to collect clustering results
    clustering_results = []

    # Iterate over each year with a progress bar
    for year in tqdm(unique_years, desc=f"Clustering Years {years_to_cluster}"):
        # Filter data for the current year
        year_df = filtered_df[filtered_df['year'] == year]

        # Check if there are enough company pairs to form clusters
        if year_df.empty:
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Initialize an empty undirected graph
        G = nx.Graph()

        # Add edges to the graph with 'sum_abs_diff_scaled_01' as the weight
        edges = list(zip(year_df['Company1'], year_df['Company2'], year_df['cosine_distance_scaled']))
        G.add_weighted_edges_from(edges)

        # Check if the graph has at least one edge
        if G.number_of_edges() == 0:
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Compute the Minimum Spanning Tree (MST)
        try:
            mst = nx.minimum_spanning_tree(G, weight='weight')
        except Exception as e:
            print(f"Error computing MST for year {year}: {e}")
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Threshold the MST: remove edges with weight > threshold
        try:
            edges_to_remove = [(u, v) for u, v, d in mst.edges(data=True) if d['weight'] > threshold]
            mst.remove_edges_from(edges_to_remove)
        except Exception as e:
            print(f"Error thresholding MST for year {year}: {e}")
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Find connected components (clusters) in the thresholded MST
        clusters = list(nx.connected_components(mst))

        # Assign unique cluster IDs
        cluster_dict = {}
        for idx, cluster in enumerate(clusters, start=1):
            cluster_dict[idx] = sorted(list(cluster))

        # Append the result
        clustering_results.append({'year': year, 'clusters': cluster_dict})

    # Convert the results to a DataFrame
    result_df = pd.DataFrame(clustering_results)

    return result_df



def create_cluster_dfs(df_compinfo, year_cluster_df):
    """
    Create SIC and Industry cluster DataFrames for each year.
    """
    year_SIC_cluster_df = []
    year_Industry_cluster_df = []

    for year in tqdm(sorted(df_compinfo['year'].unique()), desc="Generating Cluster DataFrames"):
        # Filter companies for the year
        year_data = df_compinfo[df_compinfo['year'] == year]

        # SIC clusters
        sic_clusters = year_data.groupby('sic_code')['__index_level_0__'].apply(list).to_dict()
        year_SIC_cluster_df.append({'year': year, 'clusters': sic_clusters})

        # Industry clusters
        industry_clusters = year_data.groupby('industry_classification')['__index_level_0__'].apply(list).to_dict()
        year_Industry_cluster_df.append({'year': year, 'clusters': industry_clusters})

    return pd.DataFrame(year_SIC_cluster_df), pd.DataFrame(year_Industry_cluster_df)


ds_compinfo = load_dataset("Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k")
df_compinfo = ds_compinfo['train'].to_pandas()
df_compinfo = df_compinfo[["cik", "year", "sic_code", "ticker", "__index_level_0__"]]
df_compinfo = df_compinfo.dropna(subset=['sic_code'])
print(f"Number of rows after dropping missing sic_code: {len(df_compinfo)}")
# Define a function to classify SIC codes into industries based on the first two digits
def classify_sic(sic_code):
    # Extract the first two digits of the SIC code
    first_two_digits = int(str(sic_code)[:2])

    # Map to industry categories
    if 1 <= first_two_digits <= 9:
        return 'Agriculture, Forestry, And Fishing'
    elif 10 <= first_two_digits <= 14:
        return 'Mining'
    elif 15 <= first_two_digits <= 17:
        return 'Construction'
    elif 20 <= first_two_digits <= 39:
        return 'Manufacturing'
    elif 40 <= first_two_digits <= 49:
        return 'Transportation, Communications, Electric, Gas, And Sanitary Services'
    elif 50 <= first_two_digits <= 51:
        return 'Wholesale Trade'
    elif 52 <= first_two_digits <= 59:
        return 'Retail Trade'
    elif 60 <= first_two_digits <= 67:
        return 'Finance, Insurance, And Real Estate'
    elif 70 <= first_two_digits <= 89:
        return 'Services'
    elif 90 <= first_two_digits <= 99:
        return 'Public Administration'
    else:
        return 'Unknown'
ds_compinfo = 0
# Apply the classification to the SIC codes in the dataset
df_compinfo['industry_classification'] = df_compinfo['sic_code'].apply(classify_sic)
process = psutil.Process(os.getpid())
print(f"Memory usage before: {process.memory_info().rss / 1024 ** 2:.2f} MB")

pairs_ds = load_dataset("v1ctor10/cos_sim_4000pca_exp")
pairs_df = pairs_ds['train'].to_pandas()

pairs_df = pairs_df.dropna(subset=['correlation']).reset_index(drop=True)
pairs_df["year"] = pairs_df["year"].astype(int)


year_SIC_cluster_df, year_Industry_cluster_df = create_cluster_dfs(df_compinfo, pairs_df)
year_SIC_cluster_df["year"] = year_SIC_cluster_df["year"].astype(int)
year_Industry_cluster_df["year"] = year_Industry_cluster_df["year"].astype(int)
year_SIC_cluster_df = year_SIC_cluster_df.sort_values(by='year').reset_index(drop=True)
year_Industry_cluster_df = year_Industry_cluster_df.sort_values(by='year').reset_index(drop=True)

# Create directory if it doesn't exist
os.makedirs("./data/Final Results", exist_ok=True)
year_SIC_cluster_df.to_pickle("./data/Final Results/year_cluster_dfSIC.pkl")
year_Industry_cluster_df.to_pickle("./data/Final Results/year_cluster_dfINDUSTRY.pkl")

global sic_avg_corr
global industry_avg_corr
sic_avg_corr = calculate_avg_correlation(pairs_df, year_SIC_cluster_df, "SIC")
industry_avg_corr = calculate_avg_correlation(pairs_df, year_Industry_cluster_df, "Industry")

sic_avg_corr.to_csv("./data/sic_avg_corr.csv", index=False)
industry_avg_corr.to_csv("./data/indus try_avg_corr.csv", index=False)

sic_p = sic_avg_corr.mean()[1]
ind_p = industry_avg_corr.mean()[1]
pop_p = pairs_df["correlation"].mean()
print(f"sic: {sic_p}, industry: {ind_p}, population: {pop_p}")

# Prepping distance metric
pairs_df['cosine_distance'] = 1 - pairs_df['cosine_similarity']

scaler = StandardScaler()
pairs_df['cosine_distance_scaled'] = scaler.fit_transform(pairs_df[['cosine_distance']])

import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
from functools import partial
from joblib import Parallel, delayed
import logging
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    filename='optuna_optimization.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Helper functions:


def calculate_avg_correlation(TO_ANALYSE_DF, cluster_df, cluster_type):
    """
    Calculate the mean correlation for each cluster type per year.

    Parameters:
    - TO_ANALYSE_DF (pd.DataFrame): Original DataFrame containing 'year', 'Company1', 'Company2', and 'correlation'.
    - cluster_df (pd.DataFrame): DataFrame with 'year' and 'clusters' columns.
    - cluster_type (str): Description of the cluster type for labeling.

    Returns:
    - pd.DataFrame: DataFrame with 'year' and average correlation for the cluster type.
    """
    avg_correlations = []

    for _, row in tqdm(cluster_df.iterrows(), desc=f"Calculating Stats for {cluster_type}", total=len(cluster_df)):
        year = row['year']
        clusters = row['clusters']
        year_data = TO_ANALYSE_DF[TO_ANALYSE_DF['year'] == year]
        cluster_stats = []

        for cluster_id, companies in clusters.items():
            if len(companies) <= 1:  # Skip clusters with only 1 company
                continue

            # Get all pairs of companies within the cluster
            cluster_pairs = year_data[
                (year_data['Company1'].isin(companies) & year_data['Company2'].isin(companies))
            ]
            # Calculate statistics for the cluster
            if not cluster_pairs.empty:
                try:
                    correlations = cluster_pairs['correlation']
                except KeyError:
                    # Handle alternative column name if 'correlation' doesn't exist
                    correlations = cluster_pairs['ActualCorrelation']
                cluster_stats.append(correlations.mean())

        # Aggregate statistics across clusters for the year
        if cluster_stats:
            avg_correlations.append({
                'year': year,
                f'{cluster_type}AvgCorrelation': sum(cluster_stats) / len(cluster_stats)
            })
        else:
            avg_correlations.append({
                'year': year,
                f'{cluster_type}AvgCorrelation': np.nan
            })

    return pd.DataFrame(avg_correlations)


def perform_clustering_per_year(
    TO_ANALYSE_DF,
    years_to_cluster,
    threshold,
    linkage_method='single'
):
    """
    Perform clustering on specified years using MST thresholding.

    Parameters:
    - pairs_df (pd.DataFrame): DataFrame containing company pairs with 'Company1', 'Company2', 'year', and 'sum_abs_diff_scaled_01'.
    - years_to_cluster (list): List of years to perform clustering on.
    - threshold (float): Threshold for forming clusters by removing edges from the MST.
    - linkage_method (str): Linkage method for hierarchical clustering ('single', 'complete', 'average', 'ward').

    Returns:
    - pd.DataFrame: DataFrame with 'year' and 'clusters' columns.
    """

    # Filter the DataFrame for the specified years
    filtered_df = TO_ANALYSE_DF[TO_ANALYSE_DF['year'].isin(years_to_cluster)]

    # Get sorted list of unique years within the specified subset
    unique_years = sorted(filtered_df['year'].unique())

    # Initialize list to collect clustering results
    clustering_results = []

    # Iterate over each year with a progress bar
    for year in tqdm(unique_years, desc=f"Clustering Years {years_to_cluster}"):
        # Filter data for the current year
        year_df = filtered_df[filtered_df['year'] == year]

        # Check if there are enough company pairs to form clusters
        if year_df.empty:
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Initialize an empty undirected graph
        G = nx.Graph()

        # Add edges to the graph with 'sum_abs_diff_scaled_01' as the weight
        edges = list(zip(year_df['Company1'], year_df['Company2'], year_df['cosine_distance_scaled']))
        G.add_weighted_edges_from(edges)

        # Check if the graph has at least one edge
        if G.number_of_edges() == 0:
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Compute the Minimum Spanning Tree (MST)
        try:
            mst = nx.minimum_spanning_tree(G, weight='weight')
        except Exception as e:
            print(f"Error computing MST for year {year}: {e}")
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Threshold the MST: remove edges with weight > threshold
        try:
            edges_to_remove = [(u, v) for u, v, d in mst.edges(data=True) if d['weight'] > threshold]
            mst.remove_edges_from(edges_to_remove)
        except Exception as e:
            print(f"Error thresholding MST for year {year}: {e}")
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Find connected components (clusters) in the thresholded MST
        clusters = list(nx.connected_components(mst))

        # Assign unique cluster IDs
        cluster_dict = {}
        for idx, cluster in enumerate(clusters, start=1):
            cluster_dict[idx] = sorted(list(cluster))

        # Append the result
        clustering_results.append({'year': year, 'clusters': cluster_dict})

    # Convert the results to a DataFrame
    result_df = pd.DataFrame(clustering_results)

    return result_df

# ------------------------------------------------------------
# 0.  PRE-COMPUTE the MeanCorr grid once and cache to disk
# ------------------------------------------------------------
def build_meancorr_grid(TO_ANALYSE_DF, pairs_df,
                        theta_grid=np.arange(-4.5, -0.9, 0.1),
                        cache="meancorr_grid.pkl",
                        per_year_folder="./data/Grid Search"):
    """
    Computes MeanCorr(year, θ) for every (year, θ) pair.

    * The full grid (years × θ) is saved to `cache`.
    * For every single year Y, a DataFrame
          Theta   MeanCorr
       is saved to   f"{per_year_folder}/{Y}_grid.pkl".
    """
    # --------------------------------------------------------------------
    # 0. Re-use cached master grid if present
    # --------------------------------------------------------------------
    if os.path.exists(cache):
        print(f"Loaded cached grid from {cache}")
        grid = pd.read_pickle(cache)
    else:
        years   = sorted(pairs_df["year"].unique())
        col_dfs = []

        for theta in tqdm(theta_grid, desc="Pre-computing MeanCorr grid"):
            cluster_df = perform_clustering_per_year(
                TO_ANALYSE_DF, years_to_cluster=years,
                threshold=theta, linkage_method="single"
            )
            mc_df = calculate_avg_correlation(
                TO_ANALYSE_DF, cluster_df, cluster_type=f"θ_{theta:.1f}"
            ).set_index("year")                          # rows = years

            col_dfs.append(
                mc_df[f"θ_{theta:.1f}AvgCorrelation"].rename(round(theta, 1))
            )

        grid = pd.concat(col_dfs, axis=1).sort_index()   # years × θ
        grid.to_pickle(cache)
        print(f"Saved master MeanCorr grid to {cache}")

    # --------------------------------------------------------------------
    # 1.  Save per-year grid-search tables
    # --------------------------------------------------------------------
    os.makedirs(per_year_folder, exist_ok=True)

    for year in grid.index:
        year_df = (grid.loc[year]
                        .reset_index()
                        .rename(columns={"index": "Theta", year: "MeanCorr"}))

        fname = os.path.join(per_year_folder, f"{year}_grid.pkl")
        year_df.to_pickle(fname)

    print(f"Saved per-year grid files to folder: {per_year_folder}")
    return grid



# ------------------------------------------------------------
# 1.  Walk-forward evaluation USING the pre-computed grid
# ------------------------------------------------------------
def rolling_fw_evaluation_grid(grid, K=5,
                               results_csv="./data/Grid Search/oos_mean_corr_by_year.csv"):
    """
    grid : DataFrame, rows = years, columns = θ, values = MeanCorr(year,θ)
    For each year t:
        * find θ that maximises the average of grid[t-K … t-1]
        * record that θ and grid[t, θ]  (OOS MeanCorr)
    """
    years       = grid.index.tolist()
    theta_vals  = grid.columns.tolist()
    oos_records = []

    # ensure output CSV has a header
    if not os.path.exists(results_csv):
        pd.DataFrame(columns=["Year","Theta","MeanCorr_OOS"]).to_csv(
            results_csv, index=False)

    for i in range(K, len(years)):
        test_year   = years[i]
        train_slice = grid.loc[years[i-K:i]]      # DataFrame (K rows × θ)

        # 1) pick θ* that maximises the TRAIN-window mean
        theta_star  = train_slice.mean(axis=0).idxmax()

        # 2) OOS MeanCorr for the test year
        mc_oos      = grid.loc[test_year, theta_star]

        # 3) store & append
        rec = {"Year": test_year,
               "Theta": theta_star,
               "MeanCorr_OOS": mc_oos}
        oos_records.append(rec)
        print(f"{test_year}: θ*={theta_star:.1f}  OOS MC={mc_oos:.3f}")

        pd.DataFrame([rec]).to_csv(
            results_csv, mode="a", header=False, index=False)

    return pd.DataFrame(oos_records)

global cluster_name
cluster_name = "C-CD" # C-CD = Cosine Distance
TO_ANALYSE_DF = pairs_df

theta_grid = np.arange(-4.5, -0.9, 0.1)          # 36 values

os.makedirs("./data/Grid Search", exist_ok=True)
mc_grid    = build_meancorr_grid(TO_ANALYSE_DF, pairs_df, theta_grid)
oos_table  = rolling_fw_evaluation_grid(mc_grid, K=5,
                                        results_csv="./data/Grid Search/oos_mean_corr_by_year.csv")

print(oos_table.head())
print(oos_table["MeanCorr_OOS"].mean())


import os, math, pandas as pd
import matplotlib.pyplot as plt

GRID_DIR = "./data/Grid Search"

# collect the yearly grid files
files = [f for f in os.listdir(GRID_DIR) if f.endswith("_grid.pkl")]
years = sorted(int(f.split("_")[0]) for f in files)

n      = len(years)
ncols  = 5
nrows  = math.ceil(n / ncols)

fig, axs = plt.subplots(nrows, ncols,
                        figsize=(ncols*3, nrows*2.5),
                        sharey=True)

for ax, y in zip(axs.flat, years):
    df = pd.read_pickle(f"{GRID_DIR}/{y}_grid.pkl")
    ax.plot(df["Theta"], df["MeanCorr"], marker="o", ms=3)
    ax.set_title(str(y), fontsize=8)
    ax.set_xticks([-4.5,-3.5,-2.5,-1.5])
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(True, lw=0.3)

# hide empty subplots
for ax in axs.flat[n:]:
    ax.axis("off")

fig.suptitle("MeanCorr vs. cut-off θ  (one panel per year)",
             fontsize=12, y=0.995)
fig.tight_layout()
plt.show()

os.makedirs("./images/", exist_ok=True)
fig.savefig("./images/per_year_grid_search_smoothness.png", dpi=300, bbox_inches="tight")

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 5))

# Theta line
ax1.plot(oos_table["Year"], oos_table["Theta"], marker='o', color='tab:blue', label='Theta (cut-off)')
ax1.set_xlabel("Year")
ax1.set_ylabel("Theta (cut-off)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(oos_table["Year"])  # Ensure integer x-axis ticks

# MeanCorr_OOS on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(oos_table["Year"], oos_table["MeanCorr_OOS"], marker='s', color='tab:red', label='MeanCorr_OOS')
ax2.set_ylabel("MeanCorr_OOS", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title("Theta and MeanCorr over Years")
fig.tight_layout()
plt.grid(True)
plt.show()

fig.savefig("./images/grid_search_param_smoothness.png", dpi=300, bbox_inches="tight")


import pandas as pd
from tqdm import tqdm
import networkx as nx


# Define a modified clustering function that takes per-year thresholds
def perform_year_specific_clustering(TO_ANALYSE_DF, oos_table):
    clustering_results = []

    for _, row in tqdm(oos_table.iterrows(), total=len(oos_table), desc="Clustering with Year-Specific Thresholds"):
        year = row['Year']
        threshold = row['Theta']
        year_df = TO_ANALYSE_DF[TO_ANALYSE_DF['year'] == year]

        if year_df.empty:
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        G = nx.Graph()
        edges = list(zip(year_df['Company1'], year_df['Company2'], year_df['cosine_distance_scaled']))
        G.add_weighted_edges_from(edges)

        if G.number_of_edges() == 0:
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        try:
            mst = nx.minimum_spanning_tree(G, weight='weight')
            edges_to_remove = [(u, v) for u, v, d in mst.edges(data=True) if d['weight'] > threshold]
            mst.remove_edges_from(edges_to_remove)
        except Exception as e:
            print(f"Error for year {year}: {e}")
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        clusters = list(nx.connected_components(mst))
        cluster_dict = {idx: sorted(list(cluster)) for idx, cluster in enumerate(clusters, start=1)}
        clustering_results.append({'year': year, 'clusters': cluster_dict})

    return pd.DataFrame(clustering_results)

rollingCD_cluster_df = perform_year_specific_clustering(TO_ANALYSE_DF, oos_table)
rollingCD_cluster_df["year"] = rollingCD_cluster_df["year"].astype(int)

rollingCD_cluster_df.to_pickle("./data/Final Results/year_cluster_dfrollingCD.pkl")

rollingCD_cluster_df = pd.read_pickle(f"./data/Final Results/year_cluster_dfrollingCD.pkl")
year_cluster_BERT = pd.read_pickle(f"./data/Final Results/year_cluster_dfBERT.pkl")
year_cluster_SBERT = pd.read_pickle(f"./data/Final Results/year_cluster_dfSBERT.pkl")
year_cluster_PALM = pd.read_pickle(f"./data/Final Results/year_cluster_dfPaLM-gecko.pkl")

custom_cluster_avg_corr_rollingCD = calculate_avg_correlation(TO_ANALYSE_DF, rollingCD_cluster_df, "CustomClusterRollingCD")
custom_cluster_avg_corr_BERT = calculate_avg_correlation(TO_ANALYSE_DF, year_cluster_BERT, "CustomClusterBERT")
custom_cluster_avg_corr_SBERT = calculate_avg_correlation(TO_ANALYSE_DF, year_cluster_SBERT, "CustomClusterSBERT")
custom_cluster_avg_corr_PALM = calculate_avg_correlation(TO_ANALYSE_DF, year_cluster_PALM, "CustomClusterPALM")

final_results = pd.merge(custom_cluster_avg_corr_rollingCD, sic_avg_corr, on='year', how='outer')
final_results = pd.merge(final_results, custom_cluster_avg_corr_BERT, on='year', how='outer')
final_results = pd.merge(final_results, custom_cluster_avg_corr_SBERT, on='year', how='outer')
final_results = pd.merge(final_results, custom_cluster_avg_corr_PALM, on='year', how='outer')

final_results = pd.merge(final_results, industry_avg_corr, on='year', how='outer')

temp_results = final_results
temp_results = temp_results.drop([0,1,2,3,4,5,6])
print(temp_results["CustomClusterRollingCDAvgCorrelation"].mean())

## Plot
import plotly.graph_objects as go
import kaleido

# Initialize the figure
fig = go.Figure()

# Add traces for each cluster type
fig.add_trace(go.Scatter(
    x=temp_results["year"],
    y=temp_results["CustomClusterRollingCDAvgCorrelation"],
    mode='lines+markers',
    name=r"$\huge{G_{\text{CDR}} \text{ = Sparse Features (ρ̄: 0.385)   }}$",
    line=dict(color = 'rgb(0,68,136)', width=2.5, dash='solid'),
    marker=dict(symbol='triangle-up', size=10)
))
fig.add_trace(go.Scatter(
    x=temp_results["year"],
    y=temp_results["CustomClusterPALMAvgCorrelation"],
    mode='lines+markers',
    name=r"$\huge{G_{\text{PALM}} \text{ = Embedders (ρ̄: 0.219)   }}$",
    line=dict(color='rgb(187,85,102)', width=2.5, dash='solid'),
    marker=dict(symbol='triangle-up', size=10)
))

# Add SIC
fig.add_trace(go.Scatter(
    x=temp_results["year"],
    y=temp_results["SICAvgCorrelation"],
    mode='lines+markers',
    name= r"$\huge{G_{\text{SIC}} \text{ (ρ̄: 0.231)   }}$",
    line=dict(color='rgb(221,170,51)', width=2.5, dash='solid'),
    marker=dict(symbol='square', size=10)
))


fig.update_layout(

    xaxis=dict(
        title="Year",
        tickmode='linear',
        dtick=2,  # Change from dtick=1 to dtick=2 for every 2 years
        title_font=dict(size=40),
        tickfont=dict(size=30),
        range=[2000, 2020],
        gridcolor="rgba(0, 0, 0, 0.2)",    # Set grid color to black
        gridwidth=1          # Increase grid line thickness
    ),
    yaxis=dict(
        title=r"Mean Correlation",
        title_font=dict(size=40),
        tickfont=dict(size=30),
        tickformat=".2f",
        gridcolor="rgba(0, 0, 0, 0.2)",  # Set grid color to black
        gridwidth=1          # Increase grid line thickness
    ),
    legend=dict(
        orientation='h',       # Horizontal legend
        x=0.0,                 # Start from the left
        y=0.97,                 # Place it above the plot area
        xanchor='left',
        yanchor='bottom',
        bgcolor="rgba(255,255,255,0.4)",
        bordercolor="black",
    ),
    template="plotly_white",
    margin=dict(t=150),
    width=1300,  # Adjust width
    height=900  # Adjust height
)
# Add annotations for all lines
fig.add_annotation(
    x=2009-1.5,
    y=temp_results["CustomClusterRollingCDAvgCorrelation"][(2009-1993)+1]+0.028,
    text=r"$\LARGE{Sparse \; Features}$",
    showarrow=False,
    xanchor="left",
    font=dict(size=200, color="rgb(0,68,136)")
)


fig.add_annotation(
    x=2009-1.05,
    y=temp_results["CustomClusterPALMAvgCorrelation"][(2008-1993)+1]-0.015,
    text=r"$\LARGE{Embedders}$",
    showarrow=False,
    xanchor="left",
    font=dict(size=200, color="rgb(187,85,102)")
)


fig.add_annotation(
    x=2009,
    y=temp_results["SICAvgCorrelation"][(2008-1993)+1]+0.06,
    text=r"$\LARGE{SIC}$",
    showarrow=False,
    xanchor="left",
    font=dict(size=200, color="rgb(221,170,51)")
)



# Export high-resolution image
fig.write_image(f'./images/CDR_PALM_final_plot_resized.png', scale=1, width=1920, height=1080)


# Show the figure
fig.show()