"""Script to get scatter plot and correlation for proof found data set and proof not found dataset.
Here, we compare inital clause count with other feature.
We take time out and not clause processing time out and nan as they are zeros.
The plot name are self explanatory
Result is saved on result/scatter_plot directory"""

import os
import pdb
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/five_mins_time_out')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result/scatter_plot')

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

proof_found_not_found_df = merged_data_frame.loc[merged_data_frame['Resolution Result'].isin(['Proof Found', 'Time Out'])]

# # Define a dictionary for colors, sizes, and shapes
# category_styles = {
#     'Time Out': {'color': 'blue', 'size': 50, 'marker': 'o'},  # Circle marker for Time Out
#     'Proof Found': {'color': 'green', 'size': 50, 'marker': 's'}  # Square marker for Proof Found
# }
#
# # Calculate the correlation
# correlation = proof_found_not_found_df['Initial Clause Count'].corr(proof_found_not_found_df['Processed Clause Count'])
#
# # Scatter plot
# plt.figure(figsize=(10, 6))
#
# # Plot each category with its own style
# for category, style in category_styles.items():
#     subset = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == category]
#     plt.scatter(subset['Initial Clause Count'], subset['Processed Clause Count'],
#                 alpha=0.5,
#                 c=style['color'],
#                 s=style['size'],
#                 marker=style['marker'],
#                 label=category)
#
# # Title and labels
# plt.title(f'Correlation between Initial and Processed Clause Counts (r={correlation:.2f})')
# plt.xlabel('Initial Clause Count')
# plt.ylabel('Processed Clause Count')
#
# # Add a legend
# plt.legend()
#
# # Show plot
# plt.show()
# plt.close()


# Calculate the combined correlation
combined_correlation = proof_found_not_found_df['Initial Clause Count'].corr(proof_found_not_found_df['Processed Clause Count'])

# Filter for 'Proof Found' and 'Time Out' and calculate their correlations
proof_found_df = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == 'Proof Found']
proof_found_correlation = proof_found_df['Initial Clause Count'].corr(proof_found_df['Processed Clause Count'])

time_out_df = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == 'Time Out']
time_out_correlation = time_out_df['Initial Clause Count'].corr(time_out_df['Processed Clause Count'])

# Define a dictionary for colors, sizes, and shapes
category_styles = {
    'Time Out': {'color': 'blue', 'size': 50, 'marker': 'o'},  # Circle marker for Time Out
    'Proof Found': {'color': 'green', 'size': 50, 'marker': 's'}  # Square marker for Proof Found
}

# Plotting
plt.figure(figsize=(12, 8))

# Plot each category with its own style
for category, style in category_styles.items():
    subset = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == category]
    plt.scatter(subset['Initial Clause Count'], subset['Processed Clause Count'],
                alpha=0.5,
                c=style['color'],
                s=style['size'],
                marker=style['marker'],
                label=f'{category} (r={subset["Initial Clause Count"].corr(subset["Processed Clause Count"]):.2f})')

# Title and labels
plt.title(f'Correlation between Initial and Processed Clause Counts (r={combined_correlation:.2f})')
plt.xlabel('Initial Clause Count')
plt.ylabel('Processed Clause Count')

# Add a legend
plt.legend()

plt.savefig(os.path.join(OUTPUT_DIR, "proof_found_scatterplot_pcc.pdf"))
plt.close()


# Calculate the combined correlation
combined_correlation = proof_found_not_found_df['Initial Clause Count'].corr(proof_found_not_found_df['Unprocessed Clause Count'])

# Filter for 'Proof Found' and 'Time Out' and calculate their correlations
proof_found_df = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == 'Proof Found']
proof_found_correlation = proof_found_df['Initial Clause Count'].corr(proof_found_df['Unprocessed Clause Count'])

time_out_df = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == 'Time Out']
time_out_correlation = time_out_df['Initial Clause Count'].corr(time_out_df['Unprocessed Clause Count'])

# Define a dictionary for colors, sizes, and shapes
category_styles = {
    'Time Out': {'color': 'blue', 'size': 50, 'marker': 'o'},  # Circle marker for Time Out
    'Proof Found': {'color': 'green', 'size': 50, 'marker': 's'}  # Square marker for Proof Found
}

# Plotting
plt.figure(figsize=(12, 8))

# Plot each category with its own style
for category, style in category_styles.items():
    subset = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == category]
    plt.scatter(subset['Initial Clause Count'], subset['Unprocessed Clause Count'],
                alpha=0.5,
                c=style['color'],
                s=style['size'],
                marker=style['marker'],
                label=f'{category} (r={subset["Initial Clause Count"].corr(subset["Unprocessed Clause Count"]):.2f})')

# Title and labels
plt.title(f'Correlation between Initial and Unprocessed Clause Counts (r={combined_correlation:.2f})')
plt.xlabel('Initial Clause Count')
plt.ylabel('Unprocessed Clause Count')

# Add a legend
plt.legend()

plt.savefig(os.path.join(OUTPUT_DIR, "proof_found_scatterplot_ucc.pdf"))
plt.close()


# Calculate the combined correlation
combined_correlation = proof_found_not_found_df['Initial Clause Count'].corr(proof_found_not_found_df['Factor Count'])

# Filter for 'Proof Found' and 'Time Out' and calculate their correlations
proof_found_df = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == 'Proof Found']
proof_found_correlation = proof_found_df['Initial Clause Count'].corr(proof_found_df['Factor Count'])

time_out_df = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == 'Time Out']
time_out_correlation = time_out_df['Initial Clause Count'].corr(time_out_df['Factor Count'])

# Define a dictionary for colors, sizes, and shapes
category_styles = {
    'Time Out': {'color': 'blue', 'size': 50, 'marker': 'o'},  # Circle marker for Time Out
    'Proof Found': {'color': 'green', 'size': 50, 'marker': 's'}  # Square marker for Proof Found
}

# Plotting
plt.figure(figsize=(12, 8))

# Plot each category with its own style
for category, style in category_styles.items():
    subset = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == category]
    plt.scatter(subset['Initial Clause Count'], subset['Factor Count'],
                alpha=0.5,
                c=style['color'],
                s=style['size'],
                marker=style['marker'],
                label=f'{category} (r={subset["Initial Clause Count"].corr(subset["Factor Count"]):.2f})')

# Title and labels
plt.title(f'Correlation between Initial and Unprocessed Clause Counts (r={combined_correlation:.2f})')
plt.xlabel('Initial Clause Count')
plt.ylabel('Factor Count')

# Add a legend
plt.legend()

# Show plot
# plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, "proof_found_scatterplot_fc.pdf"))
plt.close()



# Calculate the combined correlation
combined_correlation = proof_found_not_found_df['Initial Clause Count'].corr(proof_found_not_found_df['Resolvent Count'])

# Filter for 'Proof Found' and 'Time Out' and calculate their correlations
proof_found_df = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == 'Proof Found']
proof_found_correlation = proof_found_df['Initial Clause Count'].corr(proof_found_df['Resolvent Count'])

time_out_df = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == 'Time Out']
time_out_correlation = time_out_df['Initial Clause Count'].corr(time_out_df['Resolvent Count'])

# Define a dictionary for colors, sizes, and shapes
category_styles = {
    'Time Out': {'color': 'blue', 'size': 50, 'marker': 'o'},  # Circle marker for Time Out
    'Proof Found': {'color': 'green', 'size': 50, 'marker': 's'}  # Square marker for Proof Found
}

# Plotting
plt.figure(figsize=(12, 8))

# Plot each category with its own style
for category, style in category_styles.items():
    subset = proof_found_not_found_df[proof_found_not_found_df['Resolution Result'] == category]
    plt.scatter(subset['Initial Clause Count'], subset['Resolvent Count'],
                alpha=0.5,
                c=style['color'],
                s=style['size'],
                marker=style['marker'],
                label=f'{category} (r={subset["Initial Clause Count"].corr(subset["Resolvent Count"]):.2f})')

# Title and labels
plt.title(f'Correlation between Initial and Processed Clause Counts (r={combined_correlation:.2f})')
plt.xlabel('Initial Clause Count')
plt.ylabel('Resolvent Count')

# Add a legend
plt.legend()

plt.savefig(os.path.join(OUTPUT_DIR, "proof_found_scatterplot_rc.pdf"))
plt.close()