"""Script to get plots for proof not found data set i.e. Time Out dataset for each five algorithm separately
We donot take two other features Clause Processing Time Out and nan because they have zero as values
 Here 3 subplots 5 algorithms is min, mean and max of features
and 5 subplots 5 algorithms subplots with min max mean of its features.
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
merged_data_frame_unique_gc5 = merged_data_frame[merged_data_frame['Algorithm'] == 'GivenClause5']
merged_data_frame_unique_sc = merged_data_frame[merged_data_frame['Algorithm'] == 'SymbolCount']
merged_data_frame_unique_r = merged_data_frame[merged_data_frame['Algorithm'] == 'Random']
merged_data_frame_unique_fifo = merged_data_frame[merged_data_frame['Algorithm'] == 'FIFO']
merged_data_frame_unique_gc2 = merged_data_frame[merged_data_frame['Algorithm'] == 'GivenClause2']

grouped_min_gc5 = merged_data_frame_unique_gc5.select_dtypes(include=[np.number]).min()
grouped_mean_gc5 = merged_data_frame_unique_gc5.select_dtypes(include=[np.number]).mean().astype(int)
grouped_max_gc5 = merged_data_frame_unique_gc5.select_dtypes(include=[np.number]).max()

grouped_min_sc = merged_data_frame_unique_sc.select_dtypes(include=[np.number]).min()
grouped_mean_sc = merged_data_frame_unique_sc.select_dtypes(include=[np.number]).mean().astype(int)
grouped_max_sc = merged_data_frame_unique_sc.select_dtypes(include=[np.number]).max()

grouped_min_r = merged_data_frame_unique_r.select_dtypes(include=[np.number]).min()
grouped_mean_r = merged_data_frame_unique_r.select_dtypes(include=[np.number]).mean().astype(int)
grouped_max_r = merged_data_frame_unique_r.select_dtypes(include=[np.number]).max()

grouped_min_fifo = merged_data_frame_unique_fifo.select_dtypes(include=[np.number]).min()
grouped_mean_fifo = merged_data_frame_unique_fifo.select_dtypes(include=[np.number]).mean().astype(int)
grouped_max_fifo = merged_data_frame_unique_fifo.select_dtypes(include=[np.number]).max()

grouped_min_gc2 = merged_data_frame_unique_gc2.select_dtypes(include=[np.number]).min()
grouped_mean_gc2 = merged_data_frame_unique_gc2.select_dtypes(include=[np.number]).mean().astype(int)
grouped_max_gc2 = merged_data_frame_unique_gc2.select_dtypes(include=[np.number]).max()

# Define the categories
categories = merged_data_frame.columns[3:-1].to_list()

# Get the mean values for 'GEO' and 'NUM'
mean_values_gc5 = grouped_mean_gc5.values
mean_values_sc = grouped_mean_sc.values
mean_values_r = grouped_mean_r.values
mean_values_fifo = grouped_mean_fifo.values
mean_values_gc2 = grouped_mean_gc2.values
min_values_gc5 = grouped_min_gc5.values
min_values_sc = grouped_min_sc.values
min_values_r = grouped_min_r.values
min_values_fifo = grouped_min_fifo.values
min_values_gc2 = grouped_min_gc2.values
max_values_gc5 = grouped_max_gc5.values
max_values_sc = grouped_max_sc.values
max_values_r = grouped_max_r.values
max_values_fifo = grouped_max_fifo.values
max_values_gc2 = grouped_max_gc2.values

# Define the width of the bars
bar_width = 0.15
inter_group_spacing = 0.35  # Space between groups
intra_group_spacing = 0.001

# Set the positions of the bars on the x-axis
r = np.arange(len(categories))
# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(50, 30))

# Plot for 'MIN' values
axs[0].bar(r - 2 * bar_width - 2 * intra_group_spacing, min_values_gc5, color='#FFC0CB', width=bar_width, edgecolor='grey', label='GC5')
axs[0].bar(r - bar_width - intra_group_spacing, min_values_sc, color='#FC8EAC', width=bar_width, edgecolor='grey', label='SC')
axs[0].bar(r, min_values_r, color='#F09898', width=bar_width, edgecolor='grey', label='R')
axs[0].bar(r + bar_width + intra_group_spacing, min_values_fifo, color='#FF1493', width=bar_width, edgecolor='grey', label='FIFO')
axs[0].bar(r + 2 * bar_width + 2 * intra_group_spacing, min_values_gc2, color='#FF8C00', width=bar_width, edgecolor='grey', label='GC2')
axs[0].set_xticks(r)
axs[0].set_xticklabels(categories, ha='center', fontsize=10)
axs[0].set_ylabel('Min Value', fontsize=40)
axs[0].set_title('Comparison of Min Values for GEO and NUM', fontsize=40)
axs[0].tick_params(axis='both', labelsize=30)
axs[0].legend(fontsize=40)

# Plot for 'MAX' values
axs[1].bar(r - 2 * bar_width - 2 * intra_group_spacing, max_values_gc5, color='#FFC0CB', width=bar_width, edgecolor='grey', label='GC5')
axs[1].bar(r - bar_width - intra_group_spacing, max_values_sc, color='#FC8EAC', width=bar_width, edgecolor='grey', label='SC')
axs[1].bar(r, max_values_r, color='#F09898', width=bar_width, edgecolor='grey', label='R')
axs[1].bar(r + bar_width + intra_group_spacing, max_values_fifo, color='#FF1493', width=bar_width, edgecolor='grey', label='FIFO')
axs[1].bar(r + 2 * bar_width + 2 * intra_group_spacing, max_values_gc2, color='#FF8C00', width=bar_width, edgecolor='grey', label='GC2')
axs[1].set_xticks(r)
axs[1].set_xticklabels(categories, ha='center', fontsize=10)
axs[1].set_ylabel('Max Value', fontsize=40)
axs[1].set_title('Comparison of Max Values for GEO and NUM', fontsize=40)
axs[1].tick_params(axis='both', labelsize=30)
axs[1].legend(fontsize=40)

# Plot for 'MEAN' values
axs[2].bar(r - 2 * bar_width - 2 * intra_group_spacing, mean_values_gc5, color='#FFC0CB', width=bar_width, edgecolor='grey', label='GC5')
axs[2].bar(r - bar_width - intra_group_spacing, mean_values_sc, color='#FC8EAC', width=bar_width, edgecolor='grey', label='SC')
axs[2].bar(r, mean_values_r, color='#F09898', width=bar_width, edgecolor='grey', label='R')
axs[2].bar(r + bar_width + intra_group_spacing, mean_values_fifo, color='#FF1493', width=bar_width, edgecolor='grey', label='FIFO')
axs[2].bar(r + 2 * bar_width + 2 * intra_group_spacing, mean_values_gc2, color='#FF8C00', width=bar_width, edgecolor='grey', label='GC2')
axs[2].set_xticks(r)
axs[2].set_xticklabels(categories, ha='center', fontsize=20)
axs[2].set_ylabel('Mean Value', fontsize=20)
axs[2].set_title('Comparison of Mean Values for GEO and NUM', fontsize=20)
axs[2].tick_params(axis='both', labelsize=20)
axs[2].legend(fontsize=20)

# Show plot for 'MAX' values
plt.tight_layout()

# Set up the matplotlib figure
# plt.figure(figsize=(18, 5))

plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_3_subplots_5_algorithms.pdf"))
# plt.show()
plt.close()

rows = len(grouped_min_gc5.index)
cols = 1
# It looks like the goal is to place the bars for 'geo' and 'num' categories side by side for each statistic.
# Let's plot them accordingly.

# Create the subplots
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 25), constrained_layout=True)
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(100 * cols, 20 * rows), constrained_layout=True)

# Set the positions of the bars on the x-axis
# bar_width = 0.4  # the width of the bars
num_features = grouped_min_gc5.index
positions = np.arange(len(num_features))  # positions for the first set of bars
positions = np.arange(3)

axes = axes.flatten()

# Colors for the bars
colors_gc5 = ['#B3D9FF', '#FFCCCC', '#B7FFB2']
colors_sc = ['#7FB2FF', '#FF9999', '#8AFF80']
colors_r = ['#4D88FF', '#FF6666', '#5DFF4D']
colors_fifo = ['#1A5EFF', '#FF3333', '#2EFF1A']
colors_gc2 = ['#0041CC', '#CC0000', '#00CC14']

# Plotting each feature with side by side bars
for i, feature in enumerate(grouped_min_gc5.index):
    # Plot 'gc5' bars
    axes[i].bar(positions - 2 * bar_width - 2 * intra_group_spacing,
                [grouped_min_gc5[feature], grouped_max_gc5[feature], grouped_mean_gc5[feature]],
                width=bar_width, color=colors_gc5, label=['Min GC5', 'Max GC5', 'Mean GC5'])
    # Plot 'sc' bars
    axes[i].bar(positions - bar_width - intra_group_spacing, [grouped_min_sc[feature], grouped_max_sc[feature], grouped_mean_sc[feature]],
                width=bar_width, color=colors_sc, label=['Min SC', 'Max SC', 'Mean SC'])
    # Plot 'r' bars
    axes[i].bar(positions, [grouped_min_r[feature], grouped_max_r[feature], grouped_mean_r[feature]],
                width=bar_width, color=colors_r, label=['Min R', 'Max R', 'Mean R'])
    # Plot 'fifo' bars
    axes[i].bar(positions + bar_width + intra_group_spacing,
                [grouped_min_fifo[feature], grouped_max_fifo[feature], grouped_mean_fifo[feature]],
                width=bar_width, color=colors_fifo, label=['Min FIFO', 'Max FIFO', 'Mean FIFO'])
    # Plot 'gc2' bars
    axes[i].bar(positions + 2 * bar_width + 2 * intra_group_spacing,
                [grouped_min_gc2[feature], grouped_max_gc2[feature], grouped_mean_gc2[feature]],
                width=bar_width, color=colors_gc2, label=['Min GC2', 'Max GC2', 'Mean GC2'])

    # Set the titles, legends, and adjust the x-ticks
    axes[i].set_title(feature, fontsize=80)
    axes[i].set_ylabel('Values', fontsize=80)
    axes[i].set_xticks(positions)
    axes[i].set_xticklabels(['Min', 'Max', 'Mean'], fontsize=80)
    # axes[i].legend(['Geo Min', 'Geo Max', 'Geo Mean', 'Num Min', 'Num Max', 'Num Mean'])

    axes[i].tick_params(axis='both', labelsize=80)

    axes[i].legend(fontsize=40)

# Hide any unused subplots
for i in range(len(num_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()

# Show the plot
plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_5_subplots_5_algorithms.pdf"))
# plt.show()

plt.close()




########### log


grouped_min_gc5 = np.log1p(merged_data_frame_unique_gc5.select_dtypes(include=[np.number]).min())
grouped_mean_gc5 = np.log1p(merged_data_frame_unique_gc5.select_dtypes(include=[np.number]).mean().astype(int))
grouped_max_gc5 = np.log1p(merged_data_frame_unique_gc5.select_dtypes(include=[np.number]).max())

grouped_min_sc = np.log1p(merged_data_frame_unique_sc.select_dtypes(include=[np.number]).min())
grouped_mean_sc = np.log1p(merged_data_frame_unique_sc.select_dtypes(include=[np.number]).mean().astype(int))
grouped_max_sc = np.log1p(merged_data_frame_unique_sc.select_dtypes(include=[np.number]).max())

grouped_min_r = np.log1p(merged_data_frame_unique_r.select_dtypes(include=[np.number]).min())
grouped_mean_r = np.log1p(merged_data_frame_unique_r.select_dtypes(include=[np.number]).mean().astype(int))
grouped_max_r = np.log1p(merged_data_frame_unique_r.select_dtypes(include=[np.number]).max())

grouped_min_fifo = np.log1p(merged_data_frame_unique_fifo.select_dtypes(include=[np.number]).min())
grouped_mean_fifo = np.log1p(merged_data_frame_unique_fifo.select_dtypes(include=[np.number]).mean().astype(int))
grouped_max_fifo = np.log1p(merged_data_frame_unique_fifo.select_dtypes(include=[np.number]).max())

grouped_min_gc2 = np.log1p(merged_data_frame_unique_gc2.select_dtypes(include=[np.number]).min())
grouped_mean_gc2 = np.log1p(merged_data_frame_unique_gc2.select_dtypes(include=[np.number]).mean().astype(int))
grouped_max_gc2 = np.log1p(merged_data_frame_unique_gc2.select_dtypes(include=[np.number]).max())

# Define the categories
categories = merged_data_frame.columns[3:-1].to_list()

# Get the mean values for 'GEO' and 'NUM'
mean_values_gc5 = grouped_mean_gc5.values
mean_values_sc = grouped_mean_sc.values
mean_values_r = grouped_mean_r.values
mean_values_fifo = grouped_mean_fifo.values
mean_values_gc2 = grouped_mean_gc2.values
min_values_gc5 = grouped_min_gc5.values
min_values_sc = grouped_min_sc.values
min_values_r = grouped_min_r.values
min_values_fifo = grouped_min_fifo.values
min_values_gc2 = grouped_min_gc2.values
max_values_gc5 = grouped_max_gc5.values
max_values_sc = grouped_max_sc.values
max_values_r = grouped_max_r.values
max_values_fifo = grouped_max_fifo.values
max_values_gc2 = grouped_max_gc2.values

# Define the width of the bars
bar_width = 0.15
inter_group_spacing = 0.35  # Space between groups
intra_group_spacing = 0.001

# Set the positions of the bars on the x-axis
r = np.arange(len(categories))
# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(50, 30))

# Plot for 'MIN' values
axs[0].bar(r - 2 * bar_width - 2 * intra_group_spacing, min_values_gc5, color='#FFC0CB', width=bar_width, edgecolor='grey', label='GC5')
axs[0].bar(r - bar_width - intra_group_spacing, min_values_sc, color='#FC8EAC', width=bar_width, edgecolor='grey', label='SC')
axs[0].bar(r, min_values_r, color='#F09898', width=bar_width, edgecolor='grey', label='R')
axs[0].bar(r + bar_width + intra_group_spacing, min_values_fifo, color='#FF1493', width=bar_width, edgecolor='grey', label='FIFO')
axs[0].bar(r + 2 * bar_width + 2 * intra_group_spacing, min_values_gc2, color='#FF8C00', width=bar_width, edgecolor='grey', label='GC2')
axs[0].set_xticks(r)
axs[0].set_xticklabels(categories, ha='center', fontsize=10)
axs[0].set_ylabel('Min Value', fontsize=40)
axs[0].set_title('Comparison of Min Values for GEO and NUM', fontsize=40)
axs[0].tick_params(axis='both', labelsize=30)
axs[0].legend(fontsize=40)

# Plot for 'MAX' values
axs[1].bar(r - 2 * bar_width - 2 * intra_group_spacing, max_values_gc5, color='#FFC0CB', width=bar_width, edgecolor='grey', label='GC5')
axs[1].bar(r - bar_width - intra_group_spacing, max_values_sc, color='#FC8EAC', width=bar_width, edgecolor='grey', label='SC')
axs[1].bar(r, max_values_r, color='#F09898', width=bar_width, edgecolor='grey', label='R')
axs[1].bar(r + bar_width + intra_group_spacing, max_values_fifo, color='#FF1493', width=bar_width, edgecolor='grey', label='FIFO')
axs[1].bar(r + 2 * bar_width + 2 * intra_group_spacing, max_values_gc2, color='#FF8C00', width=bar_width, edgecolor='grey', label='GC2')
axs[1].set_xticks(r)
axs[1].set_xticklabels(categories, ha='center', fontsize=10)
axs[1].set_ylabel('Max Value', fontsize=40)
axs[1].set_title('Comparison of Max Values for GEO and NUM', fontsize=40)
axs[1].tick_params(axis='both', labelsize=30)
axs[1].legend(fontsize=40)

# Plot for 'MEAN' values
axs[2].bar(r - 2 * bar_width - 2 * intra_group_spacing, mean_values_gc5, color='#FFC0CB', width=bar_width, edgecolor='grey', label='GC5')
axs[2].bar(r - bar_width - intra_group_spacing, mean_values_sc, color='#FC8EAC', width=bar_width, edgecolor='grey', label='SC')
axs[2].bar(r, mean_values_r, color='#F09898', width=bar_width, edgecolor='grey', label='R')
axs[2].bar(r + bar_width + intra_group_spacing, mean_values_fifo, color='#FF1493', width=bar_width, edgecolor='grey', label='FIFO')
axs[2].bar(r + 2 * bar_width + 2 * intra_group_spacing, mean_values_gc2, color='#FF8C00', width=bar_width, edgecolor='grey', label='GC2')
axs[2].set_xticks(r)
axs[2].set_xticklabels(categories, ha='center', fontsize=10)
axs[2].set_ylabel('Mean Value', fontsize=40)
axs[2].set_title('Comparison of Mean Values for GEO and NUM', fontsize=40)
axs[2].tick_params(axis='both', labelsize=30)
axs[2].legend(fontsize=40)

# Show plot for 'MAX' values
plt.tight_layout()

# Set up the matplotlib figure
# plt.figure(figsize=(18, 5))

plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_3_log_subplots_5_algorithms.pdf"))
# plt.show()
plt.close()

rows = len(grouped_min_gc5.index)
cols = 1
# It looks like the goal is to place the bars for 'geo' and 'num' categories side by side for each statistic.
# Let's plot them accordingly.

# Create the subplots
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 25), constrained_layout=True)
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(100 * cols, 20 * rows), constrained_layout=True)

# Set the positions of the bars on the x-axis
# bar_width = 0.4  # the width of the bars
num_features = grouped_min_gc5.index
positions = np.arange(len(num_features))  # positions for the first set of bars
positions = np.arange(3)

axes = axes.flatten()

# Colors for the bars
colors_gc5 = ['#B3D9FF', '#FFCCCC', '#B7FFB2']
colors_sc = ['#7FB2FF', '#FF9999', '#8AFF80']
colors_r = ['#4D88FF', '#FF6666', '#5DFF4D']
colors_fifo = ['#1A5EFF', '#FF3333', '#2EFF1A']
colors_gc2 = ['#0041CC', '#CC0000', '#00CC14']

# Plotting each feature with side by side bars
for i, feature in enumerate(grouped_min_gc5.index):
    # Plot 'gc5' bars
    axes[i].bar(positions - 2 * bar_width - 2 * intra_group_spacing,
                [grouped_min_gc5[feature], grouped_max_gc5[feature], grouped_mean_gc5[feature]],
                width=bar_width, color=colors_gc5, label=['Min GC5', 'Max GC5', 'Mean GC5'])
    # Plot 'sc' bars
    axes[i].bar(positions - bar_width - intra_group_spacing, [grouped_min_sc[feature], grouped_max_sc[feature], grouped_mean_sc[feature]],
                width=bar_width, color=colors_sc, label=['Min SC', 'Max SC', 'Mean SC'])
    # Plot 'r' bars
    axes[i].bar(positions, [grouped_min_r[feature], grouped_max_r[feature], grouped_mean_r[feature]],
                width=bar_width, color=colors_r, label=['Min R', 'Max R', 'Mean R'])
    # Plot 'fifo' bars
    axes[i].bar(positions + bar_width + intra_group_spacing,
                [grouped_min_fifo[feature], grouped_max_fifo[feature], grouped_mean_fifo[feature]],
                width=bar_width, color=colors_fifo, label=['Min FIFO', 'Max FIFO', 'Mean FIFO'])
    # Plot 'gc2' bars
    axes[i].bar(positions + 2 * bar_width + 2 * intra_group_spacing,
                [grouped_min_gc2[feature], grouped_max_gc2[feature], grouped_mean_gc2[feature]],
                width=bar_width, color=colors_gc2, label=['Min GC2', 'Max GC2', 'Mean GC2'])

    # Set the titles, legends, and adjust the x-ticks
    axes[i].set_title(feature, fontsize=80)
    axes[i].set_ylabel('Values', fontsize=80)
    axes[i].set_xticks(positions)
    axes[i].set_xticklabels(['Min', 'Max', 'Mean'], fontsize=80)
    # axes[i].legend(['Geo Min', 'Geo Max', 'Geo Mean', 'Num Min', 'Num Max', 'Num Mean'])

    axes[i].tick_params(axis='both', labelsize=80)

    axes[i].legend(fontsize=40)

# Hide any unused subplots
for i in range(len(num_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()

# Show the plot
plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_5_log_subplots_5_algorithms.pdf"))
# plt.show()

plt.close()