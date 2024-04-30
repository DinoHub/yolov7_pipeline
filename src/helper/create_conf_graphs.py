import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_optimal_threshold(csv_path, score_column):
    df = pd.read_csv(csv_path)
    max_row = df.loc[df[score_column].idxmax()]
    return max_row['Confidence Threshold']

def plot_threshold_changes(directory, score_column):
    data = []
    optimal_thresholds = []

    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)

        if os.path.isdir(subfolder_path):
            csv_path = os.path.join(subfolder_path, 'test_fbeta_conf.csv')

            if os.path.exists(csv_path):
                number_of_data = int(subfolder.split('_')[-1][5:])
                optimal_threshold = find_optimal_threshold(csv_path, score_column)

                data.append(number_of_data)
                optimal_thresholds.append(optimal_threshold)

    data.sort()
    plt.plot(data, optimal_thresholds, label=f'Optimal Threshold ({score_column})', marker='o')

    # Add labels for each data point
    for i, txt in enumerate(data):
        plt.annotate(optimal_thresholds[i], (data[i], optimal_thresholds[i]), textcoords="offset points", xytext=(0,5), ha='center')


# Specify the directory containing the subfolders
directory_path = '/mnt/c/Users/wenyi/Documents/yolov7_pipeline/main/runs/confidence_score_correlation_expts/finetune_vary-real-data'

# Specify the score column ('F1 Score' or 'F2 Score')
# score_column = 'F1 Score'

# Plotting the results
plt.figure(figsize=(10, 6))
plot_threshold_changes(directory_path, 'F1 Score')
plot_threshold_changes(directory_path, 'F2 Score')
# plot_threshold_changes(directory_path, 'Precision')
# plot_threshold_changes(directory_path, 'Recall')
plt.xlabel('Amount of Real Data Used')
plt.ylabel('Optimal Confidence Threshold')
plt.title('(Tested on DOTAv2.0) Optimal Confidence Threshold vs. Amount of Real Data Used')
plt.legend()
plt.grid(True)
plt.savefig(Path(directory_path) / "conf_fig.png")
plt.show()


# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# def find_optimal_threshold(csv_path, score_column):
#     df = pd.read_csv(csv_path)
#     max_row = df.loc[df[score_column].idxmax()]
#     return max_row['Confidence Threshold']

# def parse_subfolder_name(subfolder):
#     parts = subfolder.split('_')
#     data_amt = parts[-1].split('-real')
#     synth_data_number = int(data_amt[0][1:])
#     real_data_number = int(data_amt[1])
#     return synth_data_number, real_data_number

# def plot_threshold_changes(directory, score_column, real_data_number):
#     data = []
#     optimal_thresholds = []

#     for subfolder in os.listdir(directory):
#         subfolder_path = os.path.join(directory, subfolder)

#         if os.path.isdir(subfolder_path):
#             csv_path = os.path.join(subfolder_path, 'test_fbeta_conf.csv')

#             if os.path.exists(csv_path):
#                 synth_data_number, real_data_number_current = parse_subfolder_name(subfolder)
                
#                 # Check if the real data number matches the specified one
#                 if real_data_number_current == real_data_number:
#                     optimal_threshold = find_optimal_threshold(csv_path, score_column)

#                     data.append(synth_data_number)
#                     optimal_thresholds.append(optimal_threshold)

#     data.sort()
#     plt.plot(data, optimal_thresholds, label=f'{score_column} for {real_data_number} Real Data', marker='o')

#     # Add labels for each data point
#     for i, txt in enumerate(data):
#         plt.annotate(optimal_thresholds[i], (data[i], optimal_thresholds[i]), textcoords="offset points", xytext=(0,5), ha='center')


# # Specify the directory containing the subfolders
# directory_path = '/mnt/c/Users/wenyi/Documents/yolov7_pipeline/main/runs/confidence_score_correlation_expts/VEDAI_fine_tune'

# # Plotting the results for F1 Score with 200 Real Data
# plt.figure(figsize=(10, 6))
# plot_threshold_changes(directory_path, 'F1 Score', 200)
# plot_threshold_changes(directory_path, 'F2 Score', 200)
# plot_threshold_changes(directory_path, 'F1 Score', 1000)
# plot_threshold_changes(directory_path, 'F2 Score', 1000)
# plt.xlabel('Amount of Synthetic Data Used')
# plt.ylabel('Optimal Confidence Threshold')
# plt.title('(Tested on VEDAI Dataset) Optimal Confidence Threshold vs. Amount of Synthetic Data Used')
# plt.legend()
# plt.grid(True)
# plt.savefig(Path(directory_path) / "conf_fig.png")
# plt.show()
