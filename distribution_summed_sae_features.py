import numpy as np
import plotly.graph_objects as go
from datasets import load_dataset
import matplotlib.pyplot as plt
import argparse


P = argparse.ArgumentParser()
P.add_argument("--img_path", type=str, default="images/distribution_summed_sae_features.png", help="Path to save the image")
args = P.parse_args()



# Load the dataset
ds = load_dataset("marco-molinari/company_reports_with_features")
sample = ds['train'].to_pandas()

# Create the 'good_features' column and calculate mean
sample['good_features'] = np.hstack(sample['features'].values)
mean_sample_features = sample['good_features']

# Convert to numpy array to avoid pandas errors
sample['good_features'] = sample['good_features']
# Define the desired length
desired_length = 131072

real_array = []

for i in range(len(sample)):
    arr = np.array(sample['good_features'][i])
    if len(arr.shape) > 0 and arr.shape[0] == desired_length:
        real_array.extend(arr)

# Convert the list of arrays 
# 
#to a 2D numpy array

array_list = np.array(real_array)

# Dimension-wise sum
#final_summed_good_features = real_array.sum(axis=0)

# Dimension-wise average
#final_averaged_good_features = real_array.sum(axis=0)
final_averaged_good_features = array_list

# Clip all values greater than 5 to exactly 5
clipped_features = np.clip(final_averaged_good_features, None, 7.5)

# Find the percentage of values that were clipped
pct_clipped = 100 * np.mean(final_averaged_good_features > 7.5)

# Create histogram with clipped values
hist_values, bin_edges = np.histogram(clipped_features, bins=300, range=(0, 7.5))
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Calculate proportions
proportions = hist_values / sum(hist_values)

# Create plot
fig = go.Figure()
fig.add_trace(go.Bar(
    x=bin_centers,
    y=proportions,
    marker_color='rgb(0,68,136)',
))


# Style the plot
fig.update_layout(
    xaxis=dict(
        title="Feature Value",
        title_font=dict(size=48),
        tickfont=dict(size=40),
        gridcolor="rgba(0, 0, 0, 0.2)",
        title_standoff=44,
        showline=False,  # Remove the x-axis line
        zeroline=False   # Remove the vertical line at x=0
    ),
    yaxis=dict(
        title="Proportion",
        title_font=dict(size=48),
        tickfont=dict(size=40),
        tickformat=".2f",
        showgrid=False,
        title_standoff=80,  # Increased from 55 to add more padding
        showline=False,
        side='left',
        automargin=True,     # Ensure margin adjusts automatically
        ticksuffix="   ",  # Add spacing after tick labels
        ticklabelposition="outside right",  # Position labels outside the axis
    ),
    template="plotly_white",
    margin=dict(t=50, b=110, l=200, r=165),  # Increased left margin from 165 to 200
    width=1430,
    height=990,
    plot_bgcolor='white'  # Ensure background is white
)

fig.update_yaxes(
    range=(0, 0.5)
)

fig.update_xaxes(
    range=(0, 7.5),
    constrain='domain'
)

# Save image
# fig.write_image('feature_distribution.png', scale=1, width=1920, height=1080)


if pct_clipped > 0:
    fig.add_annotation(
        x=bin_centers[-1],
        y=proportions[-1] * 1.12,  # Position above the bar
        text=f"Proportion > 7.5: {(pct_clipped / 100):.2f}",
        showarrow=False,  # Remove the arrow
        font=dict(size=30, color="rgb(0,68,136)"),
        yshift=25  # Shift up by 20 pixels
    )
    fig.write_image(args.img_path, scale=1, width=1920, height=1080)
