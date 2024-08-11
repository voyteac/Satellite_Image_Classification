import fnmatch
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from data_processor import static_data as sd


class ModelEvaluatorHelper:
    def __init__(self):
        self.evaluated_models_results_path = sd.evaluated_models_results_path
        self.images_root_directory = sd.images_root_directory
        self.all_metrics_from_confusion_matrices = sd.all_metrics_from_confusion_matrices
        self.results_model_name_column = sd.results_model_name_column
        self.all_label_list = self.get_all_labels_based_on_image_directories()


    def find_saved_models_in_result_path(self):
        keras_files = {}
        for root, dirs, files in os.walk(self.evaluated_models_results_path):
            for file in files:
                if fnmatch.fnmatch(file, "*.keras"):
                    file_name = os.path.splitext(file)[0]
                    keras_files[file_name] = os.path.join(root, file)
        return keras_files

    def get_all_trained_model_names(self):
        all_trained_models_dict = self.find_saved_models_in_result_path()
        return list(all_trained_models_dict.keys())

    def get_all_labels_based_on_image_directories(self):
        return os.listdir(self.images_root_directory)


    def get_column_header_names(self):
        all_labels = self.all_label_list
        all_metrics = self.all_metrics_from_confusion_matrices
        return [self.results_model_name_column] + [f"{label}_{metric}" for label in all_labels for metric in all_metrics]

    def process_model_data(self, model_data):
        processed_data = []
        for label in self.all_label_list:
            processed_data.extend(self.round_metrics(model_data[label], self.all_metrics_from_confusion_matrices))
        return processed_data

    def round_metrics(self, metrics_dict, metrics, decimal_places=3):
        return [round(metrics_dict[metric], decimal_places) for metric in metrics]

    def round_model_for_score_and_metrics(self, metrics_list, decimal_places):
        rounded_metrics_list = []
        for metrics in metrics_list:
            rounded_metrics = {key: round(value, decimal_places) if isinstance(value, (int, float)) else value for
                               key, value in metrics.items()}
            rounded_metrics_list.append(rounded_metrics)
        return rounded_metrics_list



    def plot_scores_and_metrics_for_all_models(self):  ##!!!!!!!!!!!!!!!!!!!!!!
        self.evaluated_model_scores_metrics_path = sd.evaluated_model_scores_metrics_path

        load_all_models_df = pd.read_csv(self.evaluated_model_scores_metrics_path)
        load_all_models_df_only_data = load_all_models_df.drop(load_all_models_df.columns[:2], axis=1)

        higher_value_better = sd.higher_value_better
        closer_to_one_better = sd.closer_to_one_better
        lower_value_better = sd.lower_value_better
        start_column = 2
        for score in load_all_models_df_only_data:
            score_name = load_all_models_df.columns[start_column]
            plt.figure(figsize=(10, 6), num=score_name)
            x_labels = load_all_models_df[self.results_model_name_column]
            y_labels = load_all_models_df_only_data[score]
            plt.ylim(min(y_labels) - 0.1 * (max(y_labels) - min(y_labels)),
                     max(y_labels) + 0.1 * (max(y_labels) - min(y_labels)))
            plt.title(score_name)
            # print(f"Score name: {score_name}, Min value: {min(y_labels)}, Max value: {max(y_labels)}")

            if score_name in closer_to_one_better:
                distances_from_one = np.abs(y_labels - 1)
                max_distance = np.max(distances_from_one)
                norm = plt.Normalize(max_distance, 1)
                colours = [(1 - d, d, 0) for d in distances_from_one]
            else:
                colours = [(0, 1, 0), (1, 0, 0)]
                if score_name in higher_value_better:
                    colours = colours[::-1]
                norm = plt.Normalize(min(y_labels), max(y_labels))

            cmap_name = 'value_dependent'
            cm = LinearSegmentedColormap.from_list(cmap_name, colours, N=256)
            plt.bar(x_labels, y_labels, color=cm(norm(y_labels)))

            plt.grid(True, axis='y', linestyle='--', color='gray', alpha=0.5)
            plt.xticks(fontsize=8)
            plt.tight_layout()
            save_path = '..\\results\\' + score + ".png"
            plt.savefig(save_path, dpi=300, format='png')
            plt.close()
            start_column += 1
