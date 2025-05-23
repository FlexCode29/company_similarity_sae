import os, argparse
import json
import numpy as np
import pickle
import random

P = argparse.ArgumentParser()
P.add_argument("--fuzzing_scores", default="./fuz_scores")
P.add_argument("--strong_features_path", default="strong_features_clusters_random_subset.pkl")
P.add_argument("--top_1_percent_clusters", default=25)
P.add_argument("--out_image_path", default="interp_over_sparsity.png")

args = P.parse_args()

# working with whole sample


def extract_latent(file_name: str):
    file_name = file_name.split(".txt")[0]
    layer, latent = file_name.split("_latent")
    layer = layer.split(".")[-1]
    return int(latent), int(layer)
def spot_nans(x):
    return [i for i in x if i is not None]


results = {}
path = args.fuzzing_scores

for file_name in os.listdir(path):
    latent, layer = extract_latent(file_name)


    with open(os.path.join(path, file_name), "r") as f:
        data = json.load(f)

    corrects = spot_nans([prompt["correct"] for prompt in data])

    results[latent] = np.mean(corrects) * 100

#print(results)
print(np.mean(list(results.values())))

# working with the clusters

with open(args.strong_features_path, 'rb') as f:
        features_data = pickle.load(f)



freq = {key: 0 for key in results.keys()}

skip_set = set()
for cluster in features_data.keys():
    cls = features_data[cluster]
    path = args.strong_features_path
    counts = {'strong': 0, 'weak': 0}

    N = len(cls['strong_features'])
    for i in range(N):
        s = int(cls['strong_features'][i])

        if s not in freq.keys():
            skip_set.add(s)
            continue   
        freq[s] += 1

print(f"NUmber of features in clusters: {sum(freq.values())}")
print(sorted(list(freq.values()), reverse=True)[:10], len(skip_set))

# now we see if frequency is correlated with the results
# Calculate correlation between frequency and results
features = []
frequencies = []
scores = []

for feature, frequency in freq.items():
    if feature in results:
        features.append(feature)
        frequencies.append(frequency)
        scores.append(results[feature])

# Calculate correlation
correlation = np.corrcoef(frequencies, scores)[0, 1]
print(f"Correlation between frequency and results: {correlation:.4f}")

# Optional: display some summary statistics
print(f"Number of features with both frequency and score data: {len(features)}")
print(f"Average frequency: {np.mean(frequencies):.2f}")
print(f"Average score: {np.mean(scores):.2f}%")
print(f'Correlation between Frequency and Score: {correlation:.4f}')


# Calculate total number of clusters
total_clusters = len(features_data)
# Calculate total number of features
total_features = len(freq)


percentages = list(range(int(args.top_1_percent_clusters))) 

# Lists to store data for each percentage threshold
thresholds = []
avg_scores = []
feature_percentages = []
features_remaining = []

# Calculate average scores for each percentage threshold
for percent in percentages:
    threshold = (percent / 100) * total_clusters
    features_above_threshold = [feature_id for feature_id, frequency in freq.items() 
                               if frequency >= threshold and feature_id in results]
    
    if features_above_threshold:
        avg_score = np.mean([results[feature_id] for feature_id in features_above_threshold])
    else:
        avg_score = 0
    
    percentage_left = (len(features_above_threshold) / total_features) * 100
    
    thresholds.append(percent)
    avg_scores.append(avg_score)
    feature_percentages.append(percentage_left)
    features_remaining.append(features_above_threshold)
    
    # Print the percentage of features left at this percentile
    print(f"At {percent}% threshold: {percentage_left:.2f}% features remaining")

# Create the plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots with two y-axes
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces for interpretability scores
fig.add_trace(
    go.Scatter(
        x=thresholds,
        y=avg_scores,
        mode='lines+markers',
        name=r"Interpretability (%)",
        line=dict(color='rgb(0,68,136)', width=3),  # Increased line width
        marker=dict(symbol='triangle-up', size=11)  # Increased marker size
    ),
    secondary_y=False,
)

# Add traces for feature percentages
fig.add_trace(
    go.Scatter(
        x=thresholds,
        y=feature_percentages,
        mode='lines+markers',
        name=r"Features (%)",
        line=dict(color='rgb(187,85,102)', width=3),  # Increased line width
        marker=dict(symbol='square', size=11)  # Increased marker size
    ),
    secondary_y=True,
)

# Update layout
fig.update_layout(
    xaxis=dict(
        title="Clusters where Features are Important (%)",
        title_font=dict(size=48),  # Increased by 10% (from 44)
        tickfont=dict(size=40),    # Increased by 10% (from 36)
        gridcolor="rgba(0, 0, 0, 0.2)",
        gridwidth=1,
        showgrid=True,
        title_standoff=44          # Increased by 10%
    ),
    yaxis=dict(
        title="Interpretability of Features (%)",
        title_font=dict(size=48),  # Increased by 10% (from 44)
        tickfont=dict(size=40),    # Increased by 10% (from 36)
        tickformat=".2f",
        gridcolor="rgba(0, 0, 0, 0.2)",
        gridwidth=1,
        showgrid=False,
        zeroline=True,
        showline=True,
        showticklabels=True,
        title_standoff=55          # Increased by 10%
    ),
    yaxis2=dict(
        title="Features (%)",
        title_font=dict(size=48),  # Increased by 10% (from 44)
        tickfont=dict(size=40),    # Increased by 10% (from 36)
        tickformat=".2f",
        gridcolor="rgba(0, 0, 0, 0.2)",
        gridwidth=1,
        showgrid=False,
        showticklabels=True,
        title_standoff=55          # Increased by 10%
    ),
    legend=dict(
        orientation='h',
        x=0.5,
        y=1.02,
        xanchor='center',
        yanchor='bottom',
        bgcolor="rgba(255,255,255,0.4)",
        bordercolor="black",
        font=dict(size=44)         # Increased by 10% (from 40)
    ),
    template="plotly_white",
    margin=dict(t=110, b=110, l=165, r=165),  # Increased by 10%
    width=1430,                    # Increased by 10%
    height=990                     # Increased by 10%
)

# Export high-resolution image
fig.write_image(args.out_image_path, scale=1, width=1920, height=1080)

# Remove the fig.show() call to prevent browser opening
# fig.show()  # This line is removed or commented out

print(f"Number of clusters: {total_clusters}, features studies: {total_features}, top 1% interp: {avg_scores[-1]}, average interp: {avg_scores[0]}")
