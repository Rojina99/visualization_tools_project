"""Script to get plots for proof not found data set i.e. Time Out dataset for GEO NUM and FLD domain dataset
We donot take two other features Clause Processing Time Out and nan because they have zero as values
Here 3 subplots is min, mean and max of features
and 5 subplots subplots with min max mean of its features.
Similarly we also use log of these values and plot it
The plot name are self explanatory
Result is saved on result/proof_not_found directory"""

import os
import pdb
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/five_mins_time_out')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result/proof_not_found')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_start_goal_pair_heuristic(data_dir:str):
    if os.path.exists(data_dir):
        files = glob.glob(os.path.join(data_dir, "*.csv"))
        file_data_frame = pd.DataFrame()
        for file in files:
            with open(file, 'rb') as f:
                data = pd.read_csv(f)
                if os.path.basename(file).split('.')[0] == 'FIFO_resolution_multiprocessing':
                    data.insert(1, 'Algorithm', 'FIFO')
                elif os.path.basename(file).split('.')[0] == 'SymbolCount_resolution_multiprocessing':
                    data.insert(1, 'Algorithm', 'SymbolCount')
                elif os.path.basename(file).split('.')[0] == 'GivenClause5_resolution_multiprocessing':
                    data.insert(1, 'Algorithm', 'GivenClause5')
                elif os.path.basename(file).split('.')[0] == 'Random_resolution_multiprocessing':
                    data.insert(1, 'Algorithm', 'Random')
                elif os.path.basename(file).split('.')[0] == 'GivenClause2_resolution_multiprocessing':
                    data.insert(1, 'Algorithm', 'GivenClause2')
                data.insert(1, 'BaseFilename', data['Filename'].apply(lambda x: os.path.basename(x).split('.')[0]).str.split('[^A-Za-z]').str[0])
                file_data_frame = pd.concat([file_data_frame, data], ignore_index=True)
                file_data_frame['Filename'] = file_data_frame['Filename'].apply(lambda x: os.path.basename(x).split('.')[0])
    else:
        raise FileNotFoundError("File does not exist")
    return file_data_frame

merged_data_frame = load_start_goal_pair_heuristic(DATA_DIR)

count = 0
for name, group in merged_data_frame[merged_data_frame['Resolution Result'] == 'Time Out'].groupby('BaseFilename'):
    print(name, "\n", group)
    count = count + 1
    print(count)

merged_data_frame = merged_data_frame[merged_data_frame['Resolution Result'] == 'Time Out']
merged_data_frame_unique_geo = merged_data_frame[merged_data_frame['BaseFilename'] == 'GEO']
merged_data_frame_unique_num = merged_data_frame[merged_data_frame['BaseFilename'] == 'NUM']
merged_data_frame_unique_fld = merged_data_frame[merged_data_frame['BaseFilename'] == 'FLD']

grouped_min_geo = merged_data_frame_unique_geo.select_dtypes(include=[np.number]).min()
grouped_mean_geo = merged_data_frame_unique_geo.select_dtypes(include=[np.number]).mean().astype(int)
grouped_max_geo = merged_data_frame_unique_geo.select_dtypes(include=[np.number]).max()

grouped_min_num = merged_data_frame_unique_num.select_dtypes(include=[np.number]).min()
grouped_mean_num = merged_data_frame_unique_num.select_dtypes(include=[np.number]).mean().astype(int)
grouped_max_num = merged_data_frame_unique_num.select_dtypes(include=[np.number]).max()

grouped_min_fld = merged_data_frame_unique_fld.select_dtypes(include=[np.number]).min()
grouped_mean_fld = merged_data_frame_unique_fld.select_dtypes(include=[np.number]).mean().astype(int)
grouped_max_fld = merged_data_frame_unique_fld.select_dtypes(include=[np.number]).max()

# Define the categories
categories = merged_data_frame.columns[3:-1].to_list()

# Get the mean values for 'GEO' and 'NUM'
mean_values_geo = grouped_mean_geo.values
mean_values_num = grouped_mean_num.values
mean_values_fld = grouped_mean_fld.values
min_values_geo = grouped_min_geo.values
min_values_num = grouped_min_num.values
min_values_fld = grouped_min_fld.values
max_values_geo = grouped_max_geo.values
max_values_num = grouped_max_num.values
max_values_fld = grouped_max_fld.values

# Define the width of the bars
bar_width = 0.2
inter_group_spacing = 0.25  # Space between groups
intra_group_spacing = 0.01

# Set the positions of the bars on the x-axis
r = np.arange(len(categories))
# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(30, 10))

# Plot for 'MIN' values
axs[0].bar(r - bar_width - intra_group_spacing, min_values_geo, color='#FF8C00', width=bar_width, edgecolor='grey', label='GEO')
axs[0].bar(r, min_values_num, color='#FF1493', width=bar_width, edgecolor='grey', label='NUM')
axs[0].bar(r + bar_width + intra_group_spacing, min_values_fld, color='#f09898', width=bar_width, edgecolor='grey', label='FLD')
axs[0].set_xticks(r)
axs[0].set_xticklabels(categories, rotation=90, ha='center')
axs[0].set_ylabel('Min Value')
axs[0].set_title('Comparison of Min Values for GEO, NUM and FLD')
axs[0].legend()

# Plot for 'MAX' values
axs[1].bar(r - bar_width - intra_group_spacing, max_values_geo, color='#FF8C00', width=bar_width, edgecolor='grey', label='GEO')
axs[1].bar(r, max_values_num, color='#FF1493', width=bar_width, edgecolor='grey', label='NUM')
axs[1].bar(r + bar_width + intra_group_spacing, max_values_fld, color='#f09898', width=bar_width, edgecolor='grey', label='FLD')
axs[1].set_xticks(r)
axs[1].set_xticklabels(categories, rotation=90, ha='center')
axs[1].set_ylabel('Max Value')
axs[1].set_title('Comparison of Max Values for GEO, NUM and FLD')
axs[1].legend()

# Plot for 'MEAN' values
axs[2].bar(r - bar_width - intra_group_spacing, mean_values_geo, color='#FF8C00', width=bar_width, edgecolor='grey', label='GEO')
axs[2].bar(r, mean_values_num, color='#FF1493', width=bar_width, edgecolor='grey', label='NUM')
axs[2].bar(r + bar_width + intra_group_spacing, mean_values_fld, color='#f09898', width=bar_width, edgecolor='grey', label='FLD')
axs[2].set_xticks(r)
axs[2].set_xticklabels(categories, rotation=90, ha='center')
axs[2].set_ylabel('Mean Value')
axs[2].set_title('Comparison of Mean Values for GEO, NUM and FLD')
axs[2].legend()

# Show plot for 'MAX' values
plt.tight_layout()

# Set up the matplotlib figure
# plt.figure(figsize=(18, 5))

plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_3_subplots_geo_num_fld.pdf"))
# plt.show()
plt.close()

rows = 2
cols = (len(grouped_min_geo.index) + rows - 1) // rows
# It looks like the goal is to place the bars for 'geo' and 'num' categories side by side for each statistic.
# Let's plot them accordingly.

# Create the subplots
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 25), constrained_layout=True)
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20 * cols, 10 * rows), constrained_layout=True)

# Set the positions of the bars on the x-axis
# bar_width = 0.4  # the width of the bars
num_features = grouped_min_geo.index
positions = np.arange(len(num_features))  # positions for the first set of bars
positions = np.arange(3)

axes = axes.flatten()

# Colors for the bars
colors_geo = ['skyblue', 'lightcoral', 'lightgreen']
colors_num = ['#6495ED', 'tomato', '#32CD32']
colors_fld = ['#0000FF', '#CD5C5C', '#228B22']

# Plotting each feature with side by side bars
for i, feature in enumerate(grouped_min_geo.index):
    # Plot 'geo' bars
    axes[i].bar(positions - bar_width - intra_group_spacing, [grouped_min_geo[feature], grouped_max_geo[feature], grouped_mean_geo[feature]],
                width=bar_width, color=colors_geo, label=['Min Geo', 'Max Geo', 'Mean Geo'])
    # Plot 'num' bars
    axes[i].bar(positions, [grouped_min_num[feature], grouped_max_num[feature], grouped_mean_num[feature]],
                width=bar_width, color=colors_num, label=['Min Num', 'Max Num', 'Mean Num'])
    # Plot 'fld' bars
    axes[i].bar(positions + bar_width + intra_group_spacing,
                [grouped_min_fld[feature], grouped_max_fld[feature], grouped_mean_fld[feature]],
                width=bar_width, color=colors_fld, label=['Min FLD', 'Max FLD', 'Mean FLD'])

    # Set the titles, legends, and adjust the x-ticks
    axes[i].set_title(feature, fontsize=30)
    axes[i].set_ylabel('Values', fontsize=30)
    axes[i].set_xticks(positions)
    axes[i].set_xticklabels(['Min', 'Max', 'Mean'], fontsize=30)
    axes[i].tick_params(axis='both', labelsize=25)
    # axes[i].legend(['Geo Min', 'Geo Max', 'Geo Mean', 'Num Min', 'Num Max', 'Num Mean'])
    axes[i].legend(fontsize=25)

# Hide any unused subplots
for i in range(len(num_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()

# Show the plot
plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_5_subplots_geo_num_fld.pdf"))
# plt.show()

plt.close()




grouped_min_geo_log = np.log1p(merged_data_frame_unique_geo.select_dtypes(include=[np.number]).min())
grouped_mean_geo_log = np.log1p(merged_data_frame_unique_geo.select_dtypes(include=[np.number]).mean().astype(int))
grouped_max_geo_log = np.log1p(merged_data_frame_unique_geo.select_dtypes(include=[np.number]).max())

grouped_min_num_log = np.log1p(merged_data_frame_unique_num.select_dtypes(include=[np.number]).min())
grouped_mean_num_log = np.log1p(merged_data_frame_unique_num.select_dtypes(include=[np.number]).mean().astype(int))
grouped_max_num_log = np.log1p(merged_data_frame_unique_num.select_dtypes(include=[np.number]).max())

grouped_min_fld_log = np.log1p(merged_data_frame_unique_fld.select_dtypes(include=[np.number]).min())
grouped_mean_fld_log = np.log1p(merged_data_frame_unique_fld.select_dtypes(include=[np.number]).mean().astype(int))
grouped_max_fld_log = np.log1p(merged_data_frame_unique_fld.select_dtypes(include=[np.number]).max())

# Define the categories
categories = merged_data_frame.columns[3:-1].to_list()

# Get the mean values for 'GEO' and 'NUM'
mean_values_geo_log = grouped_mean_geo_log.values
mean_values_num_log = grouped_mean_num_log.values
mean_values_fld_log = grouped_mean_fld_log.values
min_values_geo_log = grouped_min_geo_log.values
min_values_num_log = grouped_min_num_log.values
min_values_fld_log = grouped_min_fld_log.values
max_values_geo_log = grouped_max_geo_log.values
max_values_num_log = grouped_max_num_log.values
max_values_fld_log = grouped_max_fld_log.values

# Define the width of the bars
# bar_width = 0.4

# Set the positions of the bars on the x-axis
r = np.arange(len(categories))
# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(30, 10))


# Plot for 'MIN' values
axs[0].bar(r - bar_width - intra_group_spacing, min_values_geo_log, color='#FF8C00', width=bar_width, edgecolor='grey', label='GEO')
axs[0].bar(r, min_values_num_log, color='#FF1493', width=bar_width, edgecolor='grey', label='NUM')
axs[0].bar(r + bar_width + intra_group_spacing, min_values_fld_log, color='#f09898', width=bar_width, edgecolor='grey', label='FLD')
axs[0].set_xticks(r)
axs[0].set_xticklabels(categories, rotation=90, ha='center')
axs[0].set_ylabel('Min Value')
axs[0].set_title('Comparison of Min Values for GEO, NUM and FLD')
axs[0].legend()

# Plot for 'MAX' values
axs[1].bar(r - bar_width - intra_group_spacing, max_values_geo_log, color='#FF8C00', width=bar_width, edgecolor='grey', label='GEO')
axs[1].bar(r, max_values_num_log, color='#FF1493', width=bar_width, edgecolor='grey', label='NUM')
axs[1].bar(r + bar_width + intra_group_spacing, max_values_fld_log, color='#f09898', width=bar_width, edgecolor='grey', label='FLD')
axs[1].set_xticks(r)
axs[1].set_xticklabels(categories, rotation=90, ha='center')
axs[1].set_ylabel('Max Value')
axs[1].set_title('Comparison of Max Values for GEO, NUM and FLD')
axs[1].legend()

# Plot for 'MEAN' values
axs[2].bar(r - bar_width - intra_group_spacing, mean_values_geo_log, color='#FF8C00', width=bar_width, edgecolor='grey', label='GEO')
axs[2].bar(r, mean_values_num_log, color='#FF1493', width=bar_width, edgecolor='grey', label='NUM')
axs[2].bar(r + bar_width + intra_group_spacing, mean_values_fld_log, color='#f09898', width=bar_width, edgecolor='grey', label='FLD')
axs[2].set_xticks(r)
axs[2].set_xticklabels(categories, rotation=90, ha='center')
axs[2].set_ylabel('Mean Value')
axs[2].set_title('Comparison of Mean Values for GEO, NUM and FLD')
axs[2].legend()

# Show plot for 'MAX' values
plt.tight_layout()

# Set up the matplotlib figure
# plt.figure(figsize=(18, 5))

plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_3_subplots_log_geo_num_fld.pdf"))
# plt.show()
plt.close()

rows = 2
cols = (len(grouped_min_geo_log.index) + rows - 1) // rows
# It looks like the goal is to place the bars for 'geo' and 'num' categories side by side for each statistic.
# Let's plot them accordingly.

# Create the subplots
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 25), constrained_layout=True)
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20 * cols, 10 * rows), constrained_layout=True)

# Set the positions of the bars on the x-axis
# bar_width = 0.4  # the width of the bars
num_features = grouped_min_geo_log.index
positions = np.arange(len(num_features))  # positions for the first set of bars
positions = np.arange(3)

axes = axes.flatten()

# Colors for the bars
colors_geo = ['skyblue', 'lightcoral', 'lightgreen']
colors_num = ['#6495ED', 'tomato', '#32CD32']
colors_fld = ['#0000FF', '#CD5C5C', '#228B22']

# Plotting each feature with side by side bars
for i, feature in enumerate(grouped_min_geo_log.index):
    # Plot 'geo' bars
    axes[i].bar(positions - bar_width - intra_group_spacing, [grouped_min_geo_log[feature], grouped_max_geo_log[feature], grouped_mean_geo_log[feature]],
                width=bar_width, color=colors_geo, label=['Min Geo', 'Max Geo', 'Mean Geo'])
    # Plot 'num' bars
    axes[i].bar(positions, [grouped_min_num_log[feature], grouped_max_num_log[feature], grouped_mean_num_log[feature]],
                width=bar_width, color=colors_num, label=['Min Num', 'Max Num', 'Mean Num'])
    # Plot 'fld' bars
    axes[i].bar(positions + bar_width + intra_group_spacing,
                [grouped_min_fld_log[feature], grouped_max_fld_log[feature], grouped_mean_fld_log[feature]],
                width=bar_width, color=colors_fld, label=['Min FLD', 'Max FLD', 'Mean FLD'])

    # Set the titles, legends, and adjust the x-ticks
    axes[i].set_title(feature, fontsize=30)
    axes[i].set_ylabel('Values', fontsize=30)
    axes[i].set_xticks(positions)
    axes[i].set_xticklabels(['Min', 'Max', 'Mean'], fontsize=30)
    axes[i].tick_params(axis='both', labelsize=25)
    # axes[i].legend(['Geo Min', 'Geo Max', 'Geo Mean', 'Num Min', 'Num Max', 'Num Mean'])
    axes[i].legend(fontsize=25)

# Hide any unused subplots
for i in range(len(num_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()

# Show the plot
plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_5_subplots_log_geo_num_fld.pdf"))
# plt.show()

plt.close()
