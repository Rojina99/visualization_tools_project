"""Script to get plots for proof found data set where 3 subplots is min, mean and max of features
and 5 subplots subplots with min max mean of its features.
Similarly we also use log of these values and plot it
The plot name are self explanatory
For proof found we do not include taultologies deleted as it is zero in most of the cases
Result is svaed on result/proof_found directory"""
import os
import pdb
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/five_mins_time_out')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result/proof_found')

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
for name, group in merged_data_frame[merged_data_frame['Resolution Result'] == 'Proof Found'].groupby('BaseFilename'):
    print(name, "\n", group)
    count = count + 1
    print(count)

merged_data_frame_unique = merged_data_frame[merged_data_frame['Resolution Result'] == 'Proof Found'].drop_duplicates(subset=['Filename'], keep='first')
grouped_icc_min = merged_data_frame_unique['Initial Clause Count'].min()
grouped_icc_mean = merged_data_frame_unique['Initial Clause Count'].mean()
grouped_icc_max = merged_data_frame_unique['Initial Clause Count'].max()

grouped_pcc_min = merged_data_frame_unique['Initial Clause Count'].min()

### ensure it is same for all algotirthms like aother values for tautologies delted = 5
merged_data_frame_unique = merged_data_frame_unique.drop(columns=['Tautologies Deleted'])

grouped_min = merged_data_frame_unique.select_dtypes(include=[np.number]).min()
grouped_mean = merged_data_frame_unique.select_dtypes(include=[np.number]).mean().astype(int)
grouped_max = merged_data_frame_unique.select_dtypes(include=[np.number]).max()


# Set up the matplotlib figure
plt.figure(figsize=(18, 5))

# Plotting the minimum values
plt.subplot(1, 3, 1)  # 1 row, 3 cols, subplot 1
grouped_min.plot(kind='bar', color='blue', alpha=0.7)
plt.title('Minimum Values')
plt.ylabel('Value')
plt.xticks()

# Plotting the maximum values
plt.subplot(1, 3, 2)  # 1 row, 3 cols, subplot 2
grouped_mean.plot(kind='bar', color='red', alpha=0.7)
plt.title('Maximum Values')
plt.xticks()

# Plotting the mean values
plt.subplot(1, 3, 3)  # 1 row, 3 cols, subplot 3
grouped_max.plot(kind='bar', color='green', alpha=0.7)
plt.title('Mean Values')
plt.xticks()

# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "proof_found_3_subplots.pdf"))
# plt.show()
plt.close()

num_features = len(merged_data_frame_unique.columns[3:-1])

# Determine the layout of the subplots
rows = 2  # Number of rows for subplots
cols = (num_features + rows - 1) // rows  # Number of columns for subplots

# Set up the subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)

# Flatten the axes array for easy iteration, in case of a grid larger than the number of features
axes = axes.flatten()

# Plot each bar chart in a separate subplot
for i, col in enumerate(merged_data_frame_unique.columns[3:-1]):
    ax = axes[i]
    ax.bar('Min', grouped_min[col], color='blue', width=0.25, label='Min')
    ax.bar('Max', grouped_max[col], color='red', width=0.25, label='Max')
    ax.bar('Mean', grouped_mean[col], color='green', width=0.25, label='Mean')
    ax.set_title(col)
    ax.set_ylabel('Values')
    ax.legend()

ax = axes[-1]
ax.bar(merged_data_frame_unique.columns[3:-1], grouped_min, color='blue')
ax.set_title('Min of Features')
ax.set_xlabel('Features')
ax.set_ylabel('Min Values')
ax.tick_params(axis='x',labelsize=4)
# ax.set_xticks(ax.get_xticks()[::2])
ax.legend()

# Hide any unused subplots
# for i in range(num_features, len(axes)):
#     fig.delaxes(axes[i])

# Set common labels
# fig.text(0.9, 0.04, 'Statistics', ha='center', va='center')
# fig.text(0.06, 0.5, 'Values', ha='center', va='center', rotation='vertical')

plt.tight_layout()

# plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, "proof_found_5_subplots.pdf"))
plt.close()


# # Plot mean values
# plt.figure(figsize=(10, 6))
# plt.bar(merged_data_frame_unique.columns[3:-1], grouped_min, color='blue')
# plt.title('Min of Features')
# plt.xlabel('Features')
# plt.ylabel('Min Values')
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# plt.tight_layout()
# plt.show()
#
# plt.close()

grouped_min_log = np.log1p(merged_data_frame_unique.select_dtypes(include=[np.number]).min())
grouped_mean_log = np.log1p(merged_data_frame_unique.select_dtypes(include=[np.number]).mean().astype(int))
grouped_max_log = np.log1p(merged_data_frame_unique.select_dtypes(include=[np.number]).max())


# Set up the matplotlib figure
plt.figure(figsize=(18, 5))

# Plotting the minimum values
plt.subplot(1, 3, 1)  # 1 row, 3 cols, subplot 1
grouped_min_log.plot(kind='bar', color='blue', alpha=0.7)
plt.title('Minimum Values')
plt.ylabel('Value')
plt.xticks()

# Plotting the maximum values
plt.subplot(1, 3, 2)  # 1 row, 3 cols, subplot 2
grouped_mean_log.plot(kind='bar', color='red', alpha=0.7)
plt.title('Maximum Values')
plt.xticks()

# Plotting the mean values
plt.subplot(1, 3, 3)  # 1 row, 3 cols, subplot 3
grouped_max_log.plot(kind='bar', color='green', alpha=0.7)
plt.title('Mean Values')
plt.xticks()

# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "proof_found_3_log_subplots.pdf"))
# plt.show()
plt.close()

num_features = len(merged_data_frame_unique.columns[3:-1])

# Determine the layout of the subplots
rows = 2  # Number of rows for subplots
cols = (num_features + rows - 1) // rows  # Number of columns for subplots

# Set up the subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)

# Flatten the axes array for easy iteration, in case of a grid larger than the number of features
axes = axes.flatten()

# Plot each bar chart in a separate subplot
for i, col in enumerate(merged_data_frame_unique.columns[3:-1]):
    ax = axes[i]
    ax.bar('Min', grouped_min_log[col], color='blue', width=0.25, label='Min')
    ax.bar('Max', grouped_max_log[col], color='red', width=0.25, label='Max')
    ax.bar('Mean', grouped_mean_log[col], color='green', width=0.25, label='Mean')
    ax.set_title(col)
    ax.set_ylabel('Values')
    ax.legend()

ax = axes[-1]
ax.bar(merged_data_frame_unique.columns[3:-1], grouped_min_log, color='blue')
ax.set_title('Min of Features')
ax.set_xlabel('Features')
ax.set_ylabel('Min Values')
ax.tick_params(axis='x',labelsize=4)
# ax.set_xticks(ax.get_xticks()[::2])
ax.legend()

# Hide any unused subplots
# for i in range(num_features, len(axes)):
#     fig.delaxes(axes[i])

# Set common labels
# fig.text(0.9, 0.04, 'Statistics', ha='center', va='center')
# fig.text(0.06, 0.5, 'Values', ha='center', va='center', rotation='vertical')

plt.tight_layout()

# plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, "proof_found_5_log_subplots.pdf"))
plt.close()