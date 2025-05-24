# Interpretable Company Similarity with Sparse Autoencoders

<p align="center">
  <a href="https://arxiv.org/abs/2412.02605"><img src="https://img.shields.io/badge/arXiv-2309.12075-red.svg?style=for-the-badge"></a>
</p>

This repository contains the code accompanying the paper [Interpretable Company Similarity with Sparse Autoencoders](https://arxiv.org/abs/2412.02605).

## Installing

Please make sure to install all packages in requirements.txt
```
pip install -r requirements.txt
```

Running ``` cluster_feature_gpu.py ``` achieves significant speedups compared to running the same computation on CPU (either of the sh scripts should take around a minute to run on a cluster of 8 AMD Mi250x, which were kindly provided by [Nscale](https://www.nscale.com/) for this paper). Therefore, the code is written for a multi-GPU node (you should either use a cluster of GPUs, or modify the file to run on CPU, and the scripts to not use torchrun).

## Features distribution

You can reproduce Figure 1 as follows (keep in mind this can take some time):

```python distribution_summed_sae_features.py ```

## Running Interpretability

Use the ``` reproduce_*.sh ``` scripts to obtain data/images for table 2 and figure 3, 4, and 7. In particular ``` reproduce_rolling.sh ``` uses the rolling cutoff to construct the clusters, while ``` reproduce_base.sh ``` does not.


## Obtaining the inputs.

``` fuz_scores ``` was pupulated using https://github.com/EleutherAI/delphi.    
To construct the clusters, please refer to the ``` Clustering ``` section.   
PCA is calculated on all the features (not just the 1000 we have interpretations for). The PCA we use is available at: https://drive.google.com/file/d/1p9OgcPF1ZVtmLBNRYsMEirBiNVp3xcfO/view?usp=drive_link.

## Clustering

To use files in the Clustering folder please install ``` Clustering/requirements.txt ```.
Before running:
Unzip `cik_ticker_timeseries.pkl.zip` and place the `cik_ticker_timeseries.pkl` file inside `Clustering/data/cointegration/`, otherwise `Clustering/Cointegration_Pairs_Trading.py` will not run.

Tables and Figures Reproducibility:
1. Figure 2 refers to `Clustering/images/CD_PALM_final_plot_resized.png`, and can be reproduced by running `Clustering/GCD_Clustering_SAEs.py`
2. Data From Table 1 can be reproduced by running `Clustering/GCD_Clustering_SAEs.py`, `Clustering/GCDR_Clustering_SAEs.py` and `Clustering/Cointegration_Pairs_Trading.py`.
3. Figure 5 refers to `Clustering/images/optuna_study.png`, and can be reproduced by running `Clustering/G_CD_Optuna_SAEs.py`.

## Data

We use the following datasets:
- [Company descriptions (54,275 reports)](https://huggingface.co/datasets/v1ctor10/meta_data_annual_reports_tokenized_llama3_8b_with_logged_return_matrix_with_discon)
- [Company descriptions (27,888 reports)](https://huggingface.co/datasets/Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k)
- [SAE features of each descriptions](https://huggingface.co/datasets/marco-molinari/company_reports_with_features)

The features were obtained by passing all tokenized company description trough the encoder of: https://huggingface.co/EleutherAI/sae-llama-3-8b-32x at layer 30, in particular, this is a simple example using https://github.com/EleutherAI/sae:
```
sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.30", decoder=False) 
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
outputs = model(input, output_hidden_states=True)
features = sae.encoder(outputs)
``` 
