# CompanySimSAE
Repo for "Interpretable Company Similarity with Sparse Autoencoders" paper

Before running:
Unzip `cik_ticker_timeseries.pkl.zip` and place the `cik_ticker_timeseries.pkl` file inside `Clustering/data/cointegration/`, otherwise `Clustering/Cointegration_Pairs_Trading.py` will not run.

Tables and Figures Reproducibility:
1. Figure 2 refers to `Clustering/images/CD_PALM_final_plot_resized.png`, and can be reproduced by running `Clustering/GCD_Clustering_SAEs.py`
2. Data From Table 1 can be reproduced by running `Clustering/GCD_Clustering_SAEs.py`, `Clustering/GCDR_Clustering_SAEs.py` and `Clustering/Cointegration_Pairs_Trading.py`.
3. Figure 5 refers to `Clustering/images/optuna_study.png`, and can be reproduced by running `Clustering/G_CD_Optuna_SAEs.py`.

