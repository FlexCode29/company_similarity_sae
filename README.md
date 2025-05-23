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


## Running Interpretability

Use the ``` reproduce_*.sh ``` scripts to obtain data/images for table 2 and figure 3, 4, and 7. In particular ``` reproduce_rolling.sh ``` uses the rolling cutoff to construct the clusters, while ``` reproduce_base.sh ``` does not.


## Obtaining the inputs.

``` fuz_scores ``` was pupulated using https://github.com/EleutherAI/delphi.    
To construct the clusters refer to the ``` Clustering ``` folder.   
PCA is calculated on all the features (not just the 1000 we have interpretations for). The PCA we use is available at: https://drive.google.com/file/d/1p9OgcPF1ZVtmLBNRYsMEirBiNVp3xcfO/view?usp=drive_link.

## Data

We use the following datasets:
- [Company descriptions](https://huggingface.co/datasets/Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k)
- [SAE features of each descriptions](https://huggingface.co/datasets/marco-molinari/company_reports_with_features)

The features were obtained by passing all tokenized company description trough the encoder of: https://huggingface.co/EleutherAI/sae-llama-3-8b-32x at layer 30, in particular, this can be loaded as: ``` sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.30", decoder=False) ```