import yfinance as yf
import plotly.graph_objects as go
from datasets import Dataset, DatasetDict
import pandas as pd
import psutil
import os
import joblib
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
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from tqdm import tqdm
import os
import pickle
import itertools
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import psutil
import os


CD_cluster_df = pd.read_pickle(f"./data/Final Results/year_cluster_dfC-CD.pkl")
CD_cluster_df["year"] = CD_cluster_df["year"].astype(int)
CD_cluster_df = CD_cluster_df[~CD_cluster_df['year'].isin([1993, 1994, 1995])].reset_index(drop=True)

Rolling_CD_cluster_df = pd.read_pickle(f"./data/Final Results/year_cluster_dfrollingCD.pkl")
Rolling_CD_cluster_df["year"] = Rolling_CD_cluster_df["year"].astype(int)
Rolling_CD_cluster_df = Rolling_CD_cluster_df[~Rolling_CD_cluster_df['year'].isin([1993, 1994, 1995])].reset_index(drop=True)

TPG_Cluster_df = pd.read_pickle(f"./data/Final Results/year_cluster_dfPaLM-gecko.pkl")
TPG_Cluster_df["year"] = TPG_Cluster_df["year"].astype(int)
TPG_Cluster_df = TPG_Cluster_df[~TPG_Cluster_df['year'].isin([1993, 1994, 1995])].reset_index(drop=True)

year_SIC_cluster_df = pd.read_pickle("./data/cointegration/year_SIC_cluster_mapping.pkl")
year_Industry_cluster_df = pd.read_pickle("./data/cointegration/year_Industry_cluster_mapping.pkl")
time_series_data = pd.read_pickle("./data/cointegration/cik_ticker_timeseries.pkl")
with open("./data/cointegration/index_cik_ticker_map.json", "r") as json_file:
    index_cik_ticker_map = json.load(json_file)


# Function to get price series
def get_price_series(company_index, time_series_data, start_date=None, end_date=None):
    """
    Fetch the price series for a given company index, filtered by start and end dates.
    """
    if str(company_index) not in index_cik_ticker_map:
        return None
    tickers = index_cik_ticker_map[str(company_index)]["ticker"]

    for ticker in tickers:
        ticker_data = time_series_data[time_series_data['ticker'] == ticker]
        if not ticker_data.empty:
            timeseries = ticker_data.iloc[0]['timeseries']
            timeseries = pd.Series(timeseries)
            timeseries.index = pd.to_datetime(timeseries.index)

            # Filter by date range if specified
            if start_date and end_date:
                timeseries = timeseries[start_date:end_date]
            return timeseries.sort_index()
    return None

def identify_and_save_cointegrated_pairs(
    cluster_type, cluster_df, time_series_data, year, correlation_threshold=0.95, top_n=10000000
):
    """
    Identify and rank cointegrated pairs for each cluster type by p-value, saving the top N pairs.
    Only pairs with correlation above the specified threshold are tested for cointegration.
    Clusters from the specified year and the previous two years are used.
    """
    import itertools
    from statsmodels.tsa.stattools import adfuller
    from tqdm import tqdm

    # Determine the years to include
    years_to_include = [year - i for i in range(3)]  # [year, year -1, year -2]

    # Select clusters for the specified years
    clusters_list = cluster_df.loc[cluster_df['year'].isin(years_to_include), 'clusters'].values

    # Combine clusters from different years
    combined_clusters = {}
    for clusters in clusters_list:
        for cluster_id, companies in clusters.items():
            # Create a unique key for each cluster to avoid ID conflicts
            key = f"{cluster_id}_{cluster_df.loc[cluster_df['clusters'] == clusters, 'year'].values[0]}"
            if key in combined_clusters:
                combined_clusters[key].extend(companies)
                # Remove duplicates
                combined_clusters[key] = list(set(combined_clusters[key]))
            else:
                combined_clusters[key] = companies.copy()

    cointegrated_pairs = []

    # Initialize tqdm for clusters with total number of clusters
    total_clusters = len(combined_clusters)
    cluster_iterator = tqdm(combined_clusters.items(), desc=f"Processing Clusters for {cluster_type}", total=total_clusters)

    for cluster_id, companies in cluster_iterator:
        if len(companies) < 2:
            continue

        # Generate all possible pairs within the cluster
        pair_combinations = list(itertools.combinations(companies, 2))
        total_pairs = len(pair_combinations)

        # Initialize tqdm for pairs within the current cluster
        pair_iterator = tqdm(pair_combinations, desc=f"Cluster {cluster_id}", leave=False, total=total_pairs)

        for company1, company2 in pair_iterator:
            # Fetch price series for the past three years
            start_date = f"2002-01-01"
            end_date = f"{year}-12-31"
            series1 = get_price_series(company1, time_series_data, start_date=start_date, end_date=end_date)
            series2 = get_price_series(company2, time_series_data, start_date=start_date, end_date=end_date)
            if series1 is None or series2 is None:
                continue

            # Align lengths and dates
            combined_df = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
            if len(combined_df) < 60:
                continue

            # Check if all values in series are identical
            if combined_df['series1'].nunique() <= 1 or combined_df['series2'].nunique() <= 1:
                continue

            # Calculate correlation
            correlation = combined_df['series1'].corr(combined_df['series2'])

            # Check correlation threshold
            if abs(correlation) < correlation_threshold:
                continue  # Skip pairs with low correlation

            # ADF test on the spread (difference) between the two series
            spread = combined_df['series1'] - combined_df['series2']
            adf_result = adfuller(spread)
            p_value = adf_result[1]

            if p_value < 0.05:  # p-value < 0.05 indicates stationarity (cointegration)
                cointegrated_pairs.append({
                    'Company1': company1,
                    'Company2': company2,
                    'ClusterID': cluster_id,
                    'Correlation': correlation,
                    'ADFStat': adf_result[0],
                    'p-value': p_value
                })

    print("Identified # of cointegrated pairs:", len(cointegrated_pairs))
    # Convert to DataFrame
    cointegrated_pairs_df = pd.DataFrame(cointegrated_pairs)

    # Sort by p-value and select top N pairs
    try:
        cointegrated_pairs_df = cointegrated_pairs_df.sort_values(by='p-value').head(top_n)
    except:
        print(cointegrated_pairs_df)
    # Save to CSV
    cointegrated_pairs_df.to_csv(f'cointegrated_pairs_{cluster_type}.csv', index=False)
    print(f"Saved top {top_n} cointegrated pairs for {cluster_type} to cointegrated_pairs_{cluster_type}.csv")



    return cointegrated_pairs_df

import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Position:
    def __init__(self, pair, entry_date, position_type, units1, units2, entry_price1, entry_price2):
        """
        Initialize a Position.

        Parameters:
        - pair: Tuple identifying the pair (e.g., (Company1, Company2))
        - entry_date: Date when the position was opened
        - position_type: 'long_spread' or 'short_spread'
        - units1: Units of Asset 1 (positive for buy, negative for sell)
        - units2: Units of Asset 2 (positive for buy, negative for sell)
        - entry_price1: Entry price of Asset 1
        - entry_price2: Entry price of Asset 2
        """
        self.pair = pair
        self.entry_date = entry_date
        self.position_type = position_type  # 'long_spread' or 'short_spread'
        self.units1 = units1
        self.units2 = units2
        self.entry_price1 = entry_price1
        self.entry_price2 = entry_price2
        self.exit_date = None
        self.exit_price1 = None
        self.exit_price2 = None
        self.pnl = 0

    def close(self, exit_date, exit_price1, exit_price2):
        """
        Close the position and calculate PnL.

        Parameters:
        - exit_date: Date when the position was closed
        - exit_price1: Exit price of Asset 1
        - exit_price2: Exit price of Asset 2

        Returns:
        - pnl: Profit and Loss from the position
        """
        self.exit_date = exit_date
        self.exit_price1 = exit_price1
        self.exit_price2 = exit_price2

        # Calculate PnL for Asset 1
        pnl1 = self.units1 * (self.exit_price1 - self.entry_price1)

        # Calculate PnL for Asset 2
        pnl2 = self.units2 * (self.exit_price2 - self.entry_price2)

        self.pnl = pnl1 + pnl2
        return self.pnl

    def __repr__(self):
        return (f"Position(pair={self.pair}, entry_date={self.entry_date.date()}, "
                f"type={self.position_type}, units1={self.units1:.4f}, units2={self.units2:.4f})")

class PairTrader:
    def __init__(self, capital_per_pair=10000, delta=1.0, initial_capital=100000, index_cik_ticker_map=None):
        """
        Initialize the PairTrader.

        Parameters:
        - capital_per_pair: Capital allocated per pair trade
        - delta: Entry threshold (δ) for z-score
        - initial_capital: Starting cash for the portfolio
        - index_cik_ticker_map: Mapping from company indices to tickers
        """
        self.capital_per_pair = capital_per_pair
        self.delta = delta  # Entry threshold (δ)
        self.positions = []  # All closed positions
        self.open_positions = []  # Currently open positions
        self.cumulative_pnl = 0
        self.total_trades = 0
        self.pnl_series = pd.Series(dtype=float)  # Realized PnL over time

        # Portfolio tracking
        self.initial_capital = initial_capital
        self.cash = initial_capital  # Can go negative
        self.portfolio_value_series = pd.Series(dtype=float)  # Cumulative portfolio value over time

        # Company index to ticker mapping
        self.index_cik_ticker_map = index_cik_ticker_map if index_cik_ticker_map is not None else {}

        # Configure logging within the class
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Prevent adding multiple handlers if multiple instances are created
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_price_series(self, company_index, time_series_data, start_date=None, end_date=None):
        """
        Fetch the price series for a given company index, filtered by start and end dates.
        """
        if str(company_index) not in self.index_cik_ticker_map:
            # self.logger.warning(f"Company index {company_index} not found in index_cik_ticker_map.")
            return None
        tickers = self.index_cik_ticker_map[str(company_index)]["ticker"]

        for ticker in tickers:
            ticker_data = time_series_data[time_series_data['ticker'] == ticker]
            if not ticker_data.empty:
                timeseries = ticker_data.iloc[0]['timeseries']
                timeseries = pd.Series(timeseries)
                timeseries.index = pd.to_datetime(timeseries.index)

                # Filter by date range if specified
                if start_date and end_date:
                    timeseries = timeseries[start_date:end_date]
                # self.logger.info(f"Fetched price series for ticker {ticker} (Company {company_index}).")
                return timeseries.sort_index()
        # self.logger.warning(f"No price series found for Company index {company_index} with tickers {tickers}.")
        return None

    def trade_pair(self, pair, time_series_data, start_date, end_date):
        """
        Trade a single pair over a specified period.

        Parameters:
        - pair: Dictionary with 'Company1' and 'Company2' IDs
        - time_series_data: DataFrame containing price data for all companies
        - start_date: Start date for trading
        - end_date: End date for trading
        """
        self.logger.info(f"Trading pair: {pair['Company1']} & {pair['Company2']} from {start_date} to {end_date}")

        # Fetch price series for the OOS period
        series1 = self.get_price_series(pair['Company1'], time_series_data, start_date=start_date, end_date=end_date)
        series2 = self.get_price_series(pair['Company2'], time_series_data, start_date=start_date, end_date=end_date)

        # Check if both series exist and cover the entire OOS period
        if series1 is None or series2 is None:
            # self.logger.warning(f"Missing price series for pair {pair['Company1']} & {pair['Company2']}. Skipping trade.")
            return
        if pd.Timestamp(series1.index[-1]) < pd.Timestamp("2020-12-01") or pd.Timestamp(series2.index[-1]) < pd.Timestamp("2020-12-01"):
            # One or both assets are delisted before the end of the OOS period
            # self.logger.warning(f"One or both assets in pair {pair['Company1']} & {pair['Company2']} are delisted before {end_date}. Skipping trade.")
            return

        # Align lengths and dates
        combined_df = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
        if len(combined_df) < 60:  # Minimum number of data points required
            # self.logger.warning(f"Insufficient data for pair {pair['Company1']} & {pair['Company2']}. Required: 60, Available: {len(combined_df)}. Skipping trade.")
            return

        # Perform regression on the entire OOS data to get residuals
        X = sm.add_constant(combined_df['series2'])
        model = sm.OLS(combined_df['series1'], X)
        results = model.fit()

        # Calculate residuals and z-scores
        combined_df['predicted'] = results.predict(X)
        combined_df['residuals'] = combined_df['series1'] - combined_df['predicted']
        mean = combined_df['residuals'].mean()
        std = combined_df['residuals'].std()
        combined_df['z_score'] = (combined_df['residuals'] - mean) / std

        # Log residual statistics
        # self.logger.info(f"Residual statistics for pair {pair['Company1']} & {pair['Company2']}: mean={mean:.4f}, std={std:.4f}, max={combined_df['z_score'].max():.4f}, min={combined_df['z_score'].min():.4f}")

        position = None
        last_z_score = None  # To track crossover direction

        for date, row in combined_df.iterrows():
            z_score = row['z_score']
            price1 = row['series1']
            price2 = row['series2']

            # Entry and exit conditions
            if position is None:
                # Entry conditions with crossover
                if last_z_score is not None:
                    if last_z_score < self.delta and z_score > self.delta:
                        # Crossed above +δ: Short the spread
                        units1 = -(self.capital_per_pair / 2) / price1  # Short asset 1
                        units2 = (self.capital_per_pair / 2) / price2   # Long asset 2
                        position = Position(pair=(pair['Company1'], pair['Company2']),
                                            entry_date=date, position_type='short_spread',
                                            units1=units1, units2=units2,
                                            entry_price1=price1, entry_price2=price2)
                        # Adjust cash
                        cost = (units1 * price1) + (units2 * price2)
                        self.cash += cost  # Since units1 is negative (short), adding cost
                        self.open_positions.append(position)
                        # self.logger.info(f"Opened short_spread on {date.date()} for pair {pair['Company1']} & {pair['Company2']} with units1={units1:.4f}, units2={units2:.4f}. Cash after opening: {self.cash:.2f}")
                    elif last_z_score > -self.delta and z_score < -self.delta:
                        # Crossed below -δ: Long the spread
                        units1 = (self.capital_per_pair / 2) / price1    # Long asset 1
                        units2 = -(self.capital_per_pair / 2) / price2   # Short asset 2
                        position = Position(pair=(pair['Company1'], pair['Company2']),
                                            entry_date=date, position_type='long_spread',
                                            units1=units1, units2=units2,
                                            entry_price1=price1, entry_price2=price2)
                        # Adjust cash
                        cost = (units1 * price1) + (units2 * price2)
                        self.cash += cost  # units2 is negative (short), adding cost
                        self.open_positions.append(position)
                        # self.logger.info(f"Opened long_spread on {date.date()} for pair {pair['Company1']} & {pair['Company2']} with units1={units1:.4f}, units2={units2:.4f}. Cash after opening: {self.cash:.2f}")
            else:
                # Exit conditions
                if (position.position_type == 'short_spread' and z_score <= 0) or \
                   (position.position_type == 'long_spread' and z_score >= 0):
                    # Close position when residual returns to mean
                    # self.logger.info(f"Attempting to close position: {position} on {date.date()}")
                    pnl = position.close(exit_date=date, exit_price1=price1, exit_price2=price2)
                    self.cumulative_pnl += pnl
                    self.total_trades += 1
                    self.positions.append(position)

                    # Safely remove the position
                    if position in self.open_positions:
                        self.open_positions.remove(position)
                        # self.logger.info(f"Closed position: {position} on {date.date()}. PnL: {pnl:.2f}. Cash after closing: {self.cash + (position.units1 * price1) + (position.units2 * price2):.2f}")

                    # Adjust cash
                    self.cash += (position.units1 * price1) + (position.units2 * price2)  # Proceeds from closing

                    # Update PnL series
                    self.pnl_series.loc[date] = pnl

                elif abs(z_score) >= 2 * self.delta:
                    # Stop-loss condition
                    # self.logger.info(f"Attempting to close position via stop-loss: {position} on {date.date()}")
                    pnl = position.close(exit_date=date, exit_price1=price1, exit_price2=price2)
                    self.cumulative_pnl += pnl
                    self.total_trades += 1
                    self.positions.append(position)

                    # Safely remove the position
                    if position in self.open_positions:
                        self.open_positions.remove(position)
                        # self.logger.info(f"Closed position via stop-loss: {position} on {date.date()}. PnL: {pnl:.2f}. Cash after closing: {self.cash + (position.units1 * price1) + (position.units2 * price2):.2f}")
                        # self.logger.error(f"Attempted to remove a position that was not in open_positions: {position}")
                        # self.logger.error(f"Current open_positions: {self.open_positions}")

                    # Adjust cash
                    self.cash += (position.units1 * price1) + (position.units2 * price2)  # Proceeds from closing

                    # Update PnL series
                    self.pnl_series.loc[date] = pnl

            last_z_score = z_score

            # **Daily Portfolio Value Calculation**
            # Calculate unrealized PnL from open positions
            unrealized_pnl = 0
            for pos in self.open_positions:
                current_price1 = combined_df.loc[date, 'series1']
                current_price2 = combined_df.loc[date, 'series2']
                unrealized_pnl += pos.units1 * (current_price1 - pos.entry_price1) + pos.units2 * (current_price2 - pos.entry_price2)

            # Total portfolio value is cash plus unrealized PnL
            portfolio_value = self.cash + unrealized_pnl   # variable NAV
            self.portfolio_value_series.loc[date] = portfolio_value




    def trade_pairs(self, cointegrated_pairs, time_series_data, start_date, end_date):
        """
        Trade multiple cointegrated pairs over a specified period.

        Parameters:
        - cointegrated_pairs: List of pairs (each pair is a dictionary with 'Company1' and 'Company2')
        - time_series_data: DataFrame containing price data for all companies
        - start_date: Start date for trading
        - end_date: End date for trading
        """
        # self.logger.info(f"Trading from {start_date} to {end_date}")
        for pair in tqdm(cointegrated_pairs, desc="Trading Cointegrated Pairs"):
            self.trade_pair(pair, time_series_data, start_date, end_date)

    def get_results(self, total_capital):
        """
        Retrieve the trading results.

        Parameters:
        - total_capital: Total initial capital of the portfolio

        Returns:
        - Dictionary containing cumulative PnL, total trades, cumulative return, PnL series, and cumulative portfolio value
        """
        try:
            if total_capital != 0:
                cumulative_return = self.cumulative_pnl / self.initial_capital  # Adjusted to use initial_capital
            else:
                cumulative_return = 0
        except:
            cumulative_return = 0

        # Final portfolio value calculation
        final_portfolio_value = self.initial_capital + self.cumulative_pnl

        return {
            'CumulativePnL': self.cumulative_pnl,
            'TotalTrades': self.total_trades,
            'CumulativeReturn': cumulative_return,
            'PnLSeries': self.pnl_series.sort_index(),
            'portfolio_value_series': self.portfolio_value_series,
            'CumulativePortfolioValue': final_portfolio_value  # Reflecting cumulative PnL
        }

year = 2013
all_results = []
pnl_trajectories = {}
portfolio_trajectory = {}

# --------------------------------------------
# Main Execution
# --------------------------------------------


# Define your clusters (assuming these are already defined)
cluster_dfs = {
    'CD-Cluster': CD_cluster_df,
    'Rolling_CD_Cluster': Rolling_CD_cluster_df,
    'TPG-Cluster': TPG_Cluster_df,
    'SIC': year_SIC_cluster_df,
    'Industry': year_Industry_cluster_df
}

# clusters_to_process list means that these cluster groups haven't been ran, hence not saved yet (no info on these cointegration)
# We provide these cointegration data, but you can re-process it by uncommenting the below list:
# clusters_to_process = ["CD-Cluster", "Rolling_CD_Cluster", "TPG-Cluster", "SIC", "Industry"]

# We set this as an empty list since these files already exist in ./data/cointegration/, i.e ""./data/cointegration/cointegrated_pairs_CD-Cluster"
# This speeds up the process significantly.
clusters_to_process = []

os.makedirs("./data/cointegration/Traded Clusters/", exist_ok=True)
for cluster_type, cluster_df in cluster_dfs.items():
    print("\n", cluster_type, "\n")
    cointegrated_pairs_file = f'./data/cointegration/cointegrated_pairs_{cluster_type}.csv'
    if cluster_type in clusters_to_process:
        # Re-identify and save cointegrated pairs for this cluster
        # Ensure that the function 'identify_and_save_cointegrated_pairs' is defined elsewhere
        try:
            cointegrated_pairs_df = identify_and_save_cointegrated_pairs(cluster_type, cluster_df, time_series_data, year)
            cointegrated_pairs = cointegrated_pairs_df.to_dict('records')
            print("@@@@", cointegrated_pairs, cluster_type, "@@@@@@@@")
            logger.info(f"Re-identified and saved cointegrated pairs for {cluster_type}.")
        except Exception as e:
            logger.error(f"Error identifying and saving cointegrated pairs for {cluster_type}: {e}")
            continue
    else:
        # Load existing cointegrated pairs from file
        if os.path.exists(cointegrated_pairs_file):
            try:
                cointegrated_pairs_df = pd.read_csv(cointegrated_pairs_file)
                cointegrated_pairs = cointegrated_pairs_df.to_dict('records')
            except Exception as e:
                logger.error(f"Error loading {cointegrated_pairs_file}: {e}")
                continue
            print(f"Loaded cointegrated pairs for {cluster_type} from {cointegrated_pairs_file}")
            # logger.info(f"Loaded cointegrated pairs for {cluster_type} from {cointegrated_pairs_file}")
        else:
            print(f"Cointegrated pairs file for {cluster_type} not found. Skipping.")
            # logger.warning(f"Cointegrated pairs file for {cluster_type} not found. Skipping.")
            continue  # Skip this cluster if the file doesn't exist

    # Proceed to trade the pairs using the PairTrader class
    # Optionally limit the number of pairs
    # Sort by 'ADFStat' and take top 200 pairs (ensure 'ADFStat' exists)
    try:
        filtered_pairs = [pair for pair in cointegrated_pairs if (pair['p-value'] < 0.01) and (pair['Correlation'] >0.95)]
        cointegrated_pairs = filtered_pairs
        pd.DataFrame(cointegrated_pairs).to_csv(f'./data/cointegration/Traded Clusters/NEW_cointegrated_pairs_{cluster_type}.csv', index=False)

    except KeyError:
        logger.warning(f"'Correlation' column not found in {cointegrated_pairs_file}. Skipping sorting.")
        cointegrated_pairs = cointegrated_pairs
    print(f"Number of pairs to trade: {len(cointegrated_pairs)}")

    # Initialize PairTrader
    trader = PairTrader(capital_per_pair=10000, delta=1.0, initial_capital=100000, index_cik_ticker_map=index_cik_ticker_map)  # Adjust 'initial_capital' as needed

    # Trade pairs
    trader.trade_pairs(cointegrated_pairs, time_series_data, "2014-01-01", "2020-12-31")
    print("Traded:", trader.total_trades)
    logger.info(f"Total traded pairs: {trader.total_trades}")

    # Get results
    total_capital = trader.capital_per_pair * len(cointegrated_pairs)
    results = trader.get_results(total_capital)
    number_of_cointegrated_pairs = 0
    try:
        number_of_cointegrated_pairs = len(cointegrated_pairs_df)
        number_of_total_pairs_in_this_cluster = 0
        for cluster in cluster_df[cluster_df["year"].isin([year - i for i in range(3)])]["clusters"]:
            for key, values in cluster.items():
                if len(values) > 1:
                    number_of_total_pairs_in_this_cluster +=1
    except:
        pass

    percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster = 0
    try:
        percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster = number_of_cointegrated_pairs/number_of_total_pairs_in_this_cluster
    except:
        pass

    print("percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster:", percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster)

    all_results.append({
        'ClusterType': cluster_type,
        'PnL': results['CumulativePnL'],
        'Trades': results['TotalTrades'],
        'CumulativeReturn': results['CumulativeReturn'],
        'CumulativePortfolioValue': results['CumulativePortfolioValue'],
        'percentage_of_cointegrated_pairs': percentage_of_cointegrated_pairs_to_total_number_of_pairs_in_this_cluster
    })

    pnl_trajectories[cluster_type] = results['PnLSeries']
    portfolio_trajectory[cluster_type] = results['portfolio_value_series']

def _pct_returns(value_series):
    v = value_series.sort_index()
    return v.pct_change().dropna()

def sharpe_ratio(value_series, rf_annual=0.0):
    r = _pct_returns(value_series)
    excess = r - rf_annual / 252.0
    mu, sigma = excess.mean(), excess.std()
    return np.nan if sigma == 0 else (mu / sigma) * np.sqrt(252)

def evaluate_portfolios(portfolio_trajectory, rf_annual=0.0):
    rows = []
    for name, series in portfolio_trajectory.items():
        rows.append({
            "Cluster" : name,
            "Sharpe"  : sharpe_ratio(series, rf_annual),
        })
    return pd.DataFrame(rows)

metrics = evaluate_portfolios(portfolio_trajectory)
print(metrics)