strong_features_path=data/double_cross_val_strong_features_clusters_random_subset.pkl

torchrun --standalone --nnodes=1 --nproc_per_node=8     cluster_feature_gpu.py     --clusters-pkl data/double_cross_val_year_cluster_dfC-CD.pkl     --out $strong_features_path --reference-pkl $strong_features_path  --pca-model data/global_pca_model_4000.pkl

python interp_over_sparsity.py --strong_features_path $strong_features_path --top_1_percent_clusters 31 --out_image_path images/double_cross_val_interp_over_sparsity.png

python strong_features_proportion.py --strong_features_path $strong_features_path --out_image_path images/double_cross_val_features_proportion.png