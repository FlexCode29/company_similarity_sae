import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


P = argparse.ArgumentParser()
P.add_argument("--strong_features_path", default="strong_features_clusters_random_subset.pkl")
P.add_argument("--out_image_path", default="features_proportions.png")

args = P.parse_args()

# load features
with open(args.strong_features_path, 'rb') as f:
    features_data = pickle.load(f)

# Calculate the proportion of strong features for all keys
strong_features_proportions = [len(features_data[k]['strong_features']) / 1000 for k in features_data.keys()]

# Create bin edges for deciles (0, 0.1, 0.2, ..., 1.0)
bin_edges = [i/10 for i in range(11)]  # 0 to 1 in steps of 0.1

# Create a histogram with decile bins
plt.figure(figsize=(10, 6))
plt.hist(strong_features_proportions, bins=bin_edges, alpha=0.7, color='blue', 
         edgecolor='black', weights=np.ones_like(strong_features_proportions)*100/len(strong_features_proportions))

# Highlight the median with a vertical line
median_value = np.median(strong_features_proportions)
plt.axvline(median_value, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')

plt.xlabel('Features Deemed Important (proportion)')
plt.ylabel('Clusters per Decile (%)')
plt.grid(True, alpha=0.3)
plt.xticks(bin_edges)
plt.xlim(0, 1)  # Set x-axis limit to stop at 1
plt.legend()  # Show the median in the legend
plt.tight_layout()
plt.savefig(args.out_image_path)
plt.show()

print(np.mean(strong_features_proportions))
print(np.median(strong_features_proportions))
