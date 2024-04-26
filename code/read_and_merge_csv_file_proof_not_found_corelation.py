"""Script to get scatter plot and correlation for proof not found i.e. Time Out data set against its feature.
We use time out because clause processing time out and nan are zero
The plot name are self explanatory
Result is saved on result/proof_not_found/correlation directory"""

import os
import pdb
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/five_mins_time_out')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result/proof_not_found/correlation')

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

merged_data_frame_unique = merged_data_frame[merged_data_frame['Resolution Result'] == 'Time Out'].drop_duplicates(subset=['Filename'], keep='first')

# # Calculate the correlation between Initial Clause Count and Processed Clause Count
#
# correlation = merged_data_frame_unique['Initial Clause Count'].corr(merged_data_frame_unique['Processed Clause Count'])
#
# # Plotting the correlation with seaborn
#
# plt.figure(figsize=(10, 6))
#
# sns.scatterplot(x='Initial Clause Count', y='Processed Clause Count', data=merged_data_frame_unique)
#
# plt.title(f'Correlation between Initial and Processed Clause Counts (r={correlation:.2f})')
#
# plt.show()
#
# # Calculate the correlation between Initial Clause Count and Unprocessed Clause Count
#
# correlation = merged_data_frame_unique['Initial Clause Count'].corr(merged_data_frame_unique['Unprocessed Clause Count'])
#
# # Plotting the correlation with seaborn
#
# plt.figure(figsize=(10, 6))
#
# sns.scatterplot(x='Initial Clause Count', y='Unprocessed Clause Count', data=merged_data_frame_unique)
#
# plt.title(f'Correlation between Initial and Unprocessed Clause Counts (r={correlation:.2f})')
#
# plt.show()
#
# # Calculate the correlation between Initial Clause Count and Factor Count
#
# correlation = merged_data_frame_unique['Initial Clause Count'].corr(merged_data_frame_unique['Factor Count'])
#
# # Plotting the correlation with seaborn
#
# plt.figure(figsize=(10, 6))
#
# sns.scatterplot(x='Initial Clause Count', y='Factor Count', data=merged_data_frame_unique)
#
# plt.title(f'Correlation between Initial and Factor Counts (r={correlation:.2f})')
#
# plt.show()
#
# # Calculate the correlation between Initial Clause Count and Resolvent Count
#
# correlation = merged_data_frame_unique['Initial Clause Count'].corr(merged_data_frame_unique['Resolvent Count'])
#
# # Plotting the correlation with seaborn
#
# plt.figure(figsize=(10, 6))
#
# sns.scatterplot(x='Initial Clause Count', y='Resolvent Count', data=merged_data_frame_unique)
#
# plt.title(f'Correlation between Initial and Resolvent Counts (r={correlation:.2f})')
#
# plt.show()

# Setting up the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Creates a grid of 2x2 for subplots
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust horizontal and vertical spaces between plots

# List of columns to compare with 'Initial Clause Count'
comparison_columns = ['Processed Clause Count', 'Unprocessed Clause Count', 'Factor Count', 'Resolvent Count']
titles = [
    'Correlation between Initial and Processed Clause Counts',
    'Correlation between Initial and Unprocessed Clause Counts',
    'Correlation between Initial and Factor Counts',
    'Correlation between Initial and Resolvent Counts'
]

# Loop through the comparison columns and create scatter plots
for ax, column, title in zip(axs.flatten(), comparison_columns, titles):
    # Calculate correlation
    correlation = merged_data_frame_unique['Initial Clause Count'].corr(merged_data_frame_unique[column])
    # Create scatter plot
    sns.scatterplot(x='Initial Clause Count', y=column, data=merged_data_frame_unique, ax=ax)
    # Add regression line
    sns.regplot(x='Initial Clause Count', y=column, data=merged_data_frame_unique, ci=None, scatter=False, color='blue', ax=ax)
    # Set title with correlation coefficient
    ax.set_title(f'{title} (r={correlation:.2f})')
    # Optional: Improve tick label readability
    ax.tick_params(axis='x', labelrotation=45)

# plt.show()
# Show the plot
plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_corr_icc.pdf"))
plt.close()

# Setting up the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Creates a grid of 2x2 for subplots
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust horizontal and vertical spaces between plots

# List of columns to compare with 'Initial Clause Count'
comparison_columns = ['Initial Clause Count', 'Unprocessed Clause Count', 'Factor Count', 'Resolvent Count']
titles = [
    'Correlation between Processed and Initial Clause Counts',
    'Correlation between Processed and Unprocessed Clause Counts',
    'Correlation between Processed and Factor Counts',
    'Correlation between Processed and Resolvent Counts'
]

# Loop through the comparison columns and create scatter plots
for ax, column, title in zip(axs.flatten(), comparison_columns, titles):
    # Calculate correlation
    correlation = merged_data_frame_unique['Processed Clause Count'].corr(merged_data_frame_unique[column])
    # Create scatter plot
    sns.scatterplot(x='Processed Clause Count', y=column, data=merged_data_frame_unique, ax=ax)
    # Add regression line
    sns.regplot(x='Processed Clause Count', y=column, data=merged_data_frame_unique, ci=None, scatter=False, color='blue', ax=ax)
    # Set title with correlation coefficient
    ax.set_title(f'{title} (r={correlation:.2f})')
    # Optional: Improve tick label readability
    ax.tick_params(axis='x', labelrotation=45)

plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_corr_pcc.pdf"))
plt.close()

# Setting up the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Creates a grid of 2x2 for subplots
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust horizontal and vertical spaces between plots

# List of columns to compare with 'Initial Clause Count'
comparison_columns = ['Initial Clause Count', 'Processed Clause Count', 'Factor Count', 'Resolvent Count']
titles = [
    'Correlation between Unprocessed and Initial Clause Counts',
    'Correlation between Unprocessed and Processed Clause Counts',
    'Correlation between Unprocessed and Factor Counts',
    'Correlation between Unprocessed and Resolvent Counts'
]

# Loop through the comparison columns and create scatter plots
for ax, column, title in zip(axs.flatten(), comparison_columns, titles):
    # Calculate correlation
    correlation = merged_data_frame_unique['Unprocessed Clause Count'].corr(merged_data_frame_unique[column])
    # Create scatter plot
    sns.scatterplot(x='Unprocessed Clause Count', y=column, data=merged_data_frame_unique, ax=ax)
    # Add regression line
    sns.regplot(x='Unprocessed Clause Count', y=column, data=merged_data_frame_unique, ci=None, scatter=False, color='blue', ax=ax)
    # Set title with correlation coefficient
    ax.set_title(f'{title} (r={correlation:.2f})')
    # Optional: Improve tick label readability
    ax.tick_params(axis='x', labelrotation=45)

plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_corr_ucc.pdf"))
plt.close()

# Setting up the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Creates a grid of 2x2 for subplots
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust horizontal and vertical spaces between plots

# List of columns to compare with 'Initial Clause Count'
comparison_columns = ['Initial Clause Count', 'Processed Clause Count', 'Unprocessed Clause Count', 'Resolvent Count']
titles = [
    'Correlation between Factor and Initial Clause Counts',
    'Correlation between Factor and Processed Clause Counts',
    'Correlation between Factor and Unprocessed Clause Counts',
    'Correlation between Factor and Resolvent Counts'
]

# Loop through the comparison columns and create scatter plots
for ax, column, title in zip(axs.flatten(), comparison_columns, titles):
    # Calculate correlation
    correlation = merged_data_frame_unique['Factor Count'].corr(merged_data_frame_unique[column])
    # Create scatter plot
    sns.scatterplot(x='Factor Count', y=column, data=merged_data_frame_unique, ax=ax)
    # Add regression line
    sns.regplot(x='Factor Count', y=column, data=merged_data_frame_unique, ci=None, scatter=False, color='blue', ax=ax)
    # Set title with correlation coefficient
    ax.set_title(f'{title} (r={correlation:.2f})')
    # Optional: Improve tick label readability
    ax.tick_params(axis='x', labelrotation=45)

plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_corr_fc.pdf"))
plt.close()

# Setting up the figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Creates a grid of 2x2 for subplots
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust horizontal and vertical spaces between plots

# List of columns to compare with 'Initial Clause Count'
comparison_columns = ['Initial Clause Count', 'Processed Clause Count', 'Unprocessed Clause Count', 'Factor Count']
titles = [
    'Correlation between Resolvent and Initial Clause Counts',
    'Correlation between Resolvent and Processed Clause Counts',
    'Correlation between Resolvent and Unprocessed Clause Counts',
    'Correlation between Resolvent and Factor Counts'
]

# Loop through the comparison columns and create scatter plots
for ax, column, title in zip(axs.flatten(), comparison_columns, titles):
    # Calculate correlation
    correlation = merged_data_frame_unique['Resolvent Count'].corr(merged_data_frame_unique[column])
    # Create scatter plot
    sns.scatterplot(x='Resolvent Count', y=column, data=merged_data_frame_unique, ax=ax)
    # Add regression line
    sns.regplot(x='Resolvent Count', y=column, data=merged_data_frame_unique, ci=None, scatter=False, color='blue', ax=ax)
    # Set title with correlation coefficient
    ax.set_title(f'{title} (r={correlation:.2f})')
    # Optional: Improve tick label readability
    ax.tick_params(axis='x', labelrotation=45)

# plt.show()
# Show the plot
plt.savefig(os.path.join(OUTPUT_DIR, "proof_not_found_corr_rc.pdf"))
plt.close()