import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datasets import load_dataset



P = argparse.ArgumentParser()
P.add_argument("--img_path", type=str, default="images/distribution_summed_sae_features.png", help="Path to save the image")
args = P.parse_args()


ds = load_dataset("marco-molinari/company_reports_with_features")
sample = ds['train'].to_pandas()
sample['features'] = np.hstack(sample['features'].values)
real_array = sample['features'].to_numpy()
real_array = np.array([x for x in real_array if isinstance(x, np.ndarray) and x.shape[0] == 131072])
flattened_features_on_sample = real_array.flatten()

bins = np.linspace(0, 5, 100)  # Adjust the number of bins as needed
bins = np.append(bins, np.inf)   # Include an extra bin for values >10

# Calculate weights so that the sum of the histogram heights equals 1
weights = np.ones_like(flattened_features_on_sample) / len(flattened_features_on_sample)

plt.figure(figsize=(10, 6))
plt.hist(flattened_features_on_sample, bins=bins, weights=weights, edgecolor='black', alpha=0.7)

plt.xlim(0, 5)
plt.ylim(0, 0.5)

plt.title('Distribution of Feature Activations')
plt.xlabel('Feature Activation Intensity')
plt.ylabel('Proportion of Activation Intensity Value')

plt.grid(True)
plt.show()
plt.savefig(args.img_path, dpi=300, bbox_inches='tight')
