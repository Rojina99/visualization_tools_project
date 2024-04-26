"""Initial test script you can ignore it"""
import os
import pdb
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/five_mins_time_out')

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

# merged_data_frame[pd.isna(merged_data_frame['Resolution Result'])]
# merged_data_frame['Resolution Result'].unique()
# merged_data_frame[merged_data_frame['Resolution Result'] == 'Proof Found']
# merged_data_frame[(merged_data_frame['Resolution Result'] == 'Proof Found') & (merged_data_frame['Filename'] == 'GEO257+3')]

proof_found_df = merged_data_frame[merged_data_frame['Resolution Result'] == 'Proof Found']
grouped_by_filename = proof_found_df.groupby('Filename')

for name, group in grouped_by_filename:
    print(f"Filename: {name}")
    print(group, "\n")

proof_not_found_df = merged_data_frame[merged_data_frame['Resolution Result'] != 'Proof Found']
grouped_by_filename_proof_not_found = proof_not_found_df.groupby('Filename')

count = 0
for name, group in grouped_by_filename_proof_not_found:
    print(f"Filename: {name}")
    print(group, "\n")
    count = count + 1
    print(count)

# for name, group in merged_data_frame[pd.isna(merged_data_frame['Resolution Result'])].groupby('Filename'): print(name, "\n", group)

pdb.set_trace()

count = 0
for name, group in merged_data_frame[merged_data_frame['Resolution Result'] != 'Proof Found'].groupby('BaseFilename'):
    print(name, "\n", group)
    count = count + 1
    print(count)

pdb.set_trace()

count = 0
for name, group in merged_data_frame[merged_data_frame['Resolution Result'] == 'Proof Found'].groupby('BaseFilename'):
    print(name, "\n", group)
    count = count + 1
    print(count)

pdb.set_trace()

merged_data_frame_unique = merged_data_frame[merged_data_frame['Resolution Result'] == 'Proof Found'].drop_duplicates(subset=['Filename'], keep='first')
grouped_icc_min = merged_data_frame_unique['Initial Clause Count'].min()
grouped_icc_mean = merged_data_frame_unique['Initial Clause Count'].mean()
grouped_icc_max = merged_data_frame_unique['Initial Clause Count'].max()

grouped_pcc_min = merged_data_frame_unique['Initial Clause Count'].min()

grouped_min = merged_data_frame_unique.select_dtypes(include=[np.number]).min()
grouped_mean = merged_data_frame_unique.select_dtypes(include=[np.number]).mean()
grouped_max = merged_data_frame_unique.select_dtypes(include=[np.number]).max()

pdb.set_trace()


# Set up the matplotlib figure
plt.figure(figsize=(18, 5))

# Plotting the minimum values
plt.subplot(1, 3, 1)  # 1 row, 3 cols, subplot 1
grouped_min.plot(kind='bar', color='blue', alpha=0.7)
plt.title('Minimum Values')
plt.ylabel('Value')
plt.xticks(rotation=0)

# Plotting the maximum values
plt.subplot(1, 3, 2)  # 1 row, 3 cols, subplot 2
grouped_mean.plot(kind='bar', color='red', alpha=0.7)
plt.title('Maximum Values')
plt.xticks(rotation=0)

# Plotting the mean values
plt.subplot(1, 3, 3)  # 1 row, 3 cols, subplot 3
grouped_max.plot(kind='bar', color='green', alpha=0.7)
plt.title('Mean Values')
plt.xticks(rotation=0)

# Show the plot
plt.tight_layout()
plt.show()


