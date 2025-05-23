strong_features_path=rolling_strong_features_clusters_random_subset.pkl

torchrun --standalone --nnodes=1 --nproc_per_node=8     cluster_feature_gpu.py     --clusters-pkl "rolling_year_cluster_dfCD.pkl"     --out $strong_features_path --sample-size 300  --pca-model global_pca_model_4000.pkl

python interp_over_sparsity.pt --strong_features_path $strong_features_path --top_1_percent_clusters 25 --out_image_path rolling_interp_over_sparsity.png

python strong_features_proportion.py --strong_features_path $strong_features_path --out_image_path rolling_features_proportion.png