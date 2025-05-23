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
from functools import partial
from joblib import Parallel, delayed
import logging
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import psutil
process = psutil.Process(os.getpid())
print(f"Memory usage before: {process.memory_info().rss / 1024 ** 2:.2f} MB")

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

def calculate_avg_correlation(TO_ANALYSE_DF, cluster_df, cluster_type):
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


def process_fold(fold_idx, train_years, val_years, threshold, linkage_method, pairs_df, TO_ANALYSE_DF):
    """
    Process a single fold: perform clustering on training years and calculate average correlation on validation years.

    Parameters:
    - fold_idx (int): Index of the fold.
    - train_years (list): Years to train on.
    - val_years (list): Years to validate on.
    - threshold (float): Threshold for clustering.
    - linkage_method (str): Linkage method for clustering.
    - pairs_df (pd.DataFrame): DataFrame containing company pairs.
    - TO_ANALYSE_DF (pd.DataFrame): DataFrame containing correlations.

    Returns:
    - float: Mean average correlation for the validation fold.
    """
    # Perform clustering on training years (optional, currently not used)
    train_cluster_df = perform_clustering_per_year(
        TO_ANALYSE_DF=TO_ANALYSE_DF,
        years_to_cluster=train_years,
        threshold=threshold,
        linkage_method=linkage_method
    )

    # Perform clustering on validation years
    val_cluster_df = perform_clustering_per_year(
        TO_ANALYSE_DF=TO_ANALYSE_DF,
        years_to_cluster=val_years,
        threshold=threshold,
        linkage_method=linkage_method
    )

    # Calculate average correlation for validation clusters
    val_avg_corr_df = calculate_avg_correlation(
        TO_ANALYSE_DF=TO_ANALYSE_DF,
        cluster_df=val_cluster_df,
        cluster_type=f"ValFold{fold_idx}"
    )

    # Compute the mean of average correlations for this fold
    mean_val_corr = val_avg_corr_df[f'ValFold{fold_idx}AvgCorrelation'].mean()

    # Handle cases where mean_val_corr is NaN
    if np.isnan(mean_val_corr):
        mean_val_corr = -np.inf  # Assign worst possible score

    return mean_val_corr

def optimise_cluster_parameters(TO_ANALYSE_DF, pairs_df):
    """
    Optimize clustering parameters using Optuna with temporal cross-validation and parallel processing.

    Parameters:
    - TO_ANALYSE_DF (pd.DataFrame): Original DataFrame containing 'year', 'Company1', 'Company2', and 'correlation'.
    - pairs_df (pd.DataFrame): DataFrame containing 'Company1', 'Company2', 'year', and 'sum_abs_diff_scaled_01'.

    Returns:
    - optuna.Study: The Optuna study object after optimization.
    - dict: Best parameters found by Optuna.
    """
    def objective(trial):
        # Suggest values for hyperparameters
        threshold = trial.suggest_float('threshold', -5, 2.5, step=0.01)
        linkage_method = 'single'

        # Define temporal cross-validation folds (2 folds)
        unique_years_sorted = sorted(pairs_df['year'].unique())
        n_years = len(unique_years_sorted)

        # Define two temporal splits
        fold1_train_years = unique_years_sorted[:n_years//2]
        fold1_val_years = unique_years_sorted[n_years//2:]

        fold2_train_years = unique_years_sorted[:(3*n_years)//4]
        fold2_val_years = unique_years_sorted[(3*n_years)//4:]

        folds = [
            (fold1_train_years, fold1_val_years),
            (fold2_train_years, fold2_val_years)
        ]

        # Prepare arguments for parallel processing
        tasks = [
            (idx+1, train, val, threshold, linkage_method, pairs_df, TO_ANALYSE_DF)
            for idx, (train, val) in enumerate(folds)
        ]

        # Execute folds in parallel
        fold_results = Parallel(n_jobs=-1)(
            delayed(process_fold)(fold_idx, train_years, val_years, threshold, linkage_method, pairs_df, TO_ANALYSE_DF)
            for (fold_idx, train_years, val_years, threshold, linkage_method, pairs_df, TO_ANALYSE_DF) in tasks
        )

        # Aggregate the average correlations across folds (mean)
        overall_avg_corr = np.mean(fold_results)

        # Log the parameters and the resulting correlation
        logging.info(f"Threshold: {threshold}, Linkage Method: {linkage_method}, Average Correlation: {overall_avg_corr}")

        return overall_avg_corr  # Optuna will maximize this

    # Create an Optuna study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=1))

    # Optimize the study with temporal cross-validation and parallel processing
    study.optimize(objective, n_trials=500, timeout=100000, callbacks=[save_study_callback])

    # Retrieve the best parameters
    best_params = study.best_params
    best_score = study.best_value

    print(f"\nBest parameters found by Optuna:")
    print(f"Threshold: {best_params['threshold']}")
    print(f"Linkage Method: {best_params['linkage_method']}")
    print(f"Best Average Correlation: {best_score}")

    return study, best_params



def save_study_callback(study, trial):
    """
    Callback to save trial results to CSV after each trial completes.
    """
    trial_data = {
        'number': trial.number,
        'value': trial.value,
        'state': str(trial.state)
    }

    for k, v in trial.params.items():
        trial_data[k] = v

    df = pd.DataFrame([trial_data])
    os.makedirs("./data/Clustering Optuna Study/", exist_ok=True)
    filename = f"./data/Clustering Optuna Study/{cluster_name}-cluster_study_results.csv"
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, index=False, mode='a', header=not file_exists)
    print("Trial added to CSV.")
    print(df)
    print("\n")

global cluster_name
cluster_name = "C-CD" # C-CD = Cosine Distance
TO_ANALYSE_DF = pairs_df

# Optimize clustering parameters using Optuna with temporal cross-validation and parallel processing
study, best_params = optimise_cluster_parameters(TO_ANALYSE_DF=TO_ANALYSE_DF, pairs_df=pairs_df)
filename = f"./data/Clustering Optuna Study/{cluster_name}-cluster_study_results.csv"
df_study = pd.read_csv(filename)
df_study.sort_values("value", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(df_study['threshold'], df_study['value'], width=0.01, align='center', alpha=0.75)
plt.title("Histogram of Threshold vs Value (i.e. mean corr)", fontsize=14)
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Value (i.e. mean corr)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
os.makedirs("./images/", exist_ok=True)

plt.savefig('./images/optuna_study.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()