import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pymcdm.normalizations import minmax_normalization
from data_processor import static_data as sd


class McdmCalculatorHelper:
    def __init__(self, evaluated_model_path, data_types, data_weights, is_confusion_matrix=False):
        self.evaluated_model_path = evaluated_model_path
        self.data_types = data_types
        self.data_weights = data_weights

        self.evaluated_models_results_directory = sd.evaluated_models_results_path
        self.results_model_name_column = sd.results_model_name_column

        self.evaluated_models_dataframe = self.read_csv(self.evaluated_model_path)

        self.picture_title = self.get_picture_title_for_picture(is_confusion_matrix)
        self.file_postfix = self.get_file_postifx_name_for_picture(is_confusion_matrix)

    def read_csv(self, csv_file):
        try:
            return pd.read_csv(csv_file)
        except FileNotFoundError as err:
            raise ValueError(f"File not found: {self.evaluated_model_path}") from err
        except pd.errors.ParserError as err:
            raise ValueError(f"Error parsing file: {self.evaluated_model_path}") from err

    def get_bounds(self):
        data_matrix = self.get_data_matrix()
        bounds = np.vstack((
            np.min(data_matrix, axis=0),
            np.max(data_matrix, axis=0))).T
        return bounds

    def get_data_matrix(self):
        return self.drop_model_name_from_df_data().to_numpy()

    def drop_model_name_from_df_data(self):
        first_column_with_data = 1
        return self.evaluated_models_dataframe.drop(self.evaluated_models_dataframe.columns[:first_column_with_data], axis=1)

    def get_data_types(self):
        type_list = np.array([])
        for column_name in self.drop_model_name_from_df_data():
            if column_name in self.data_types:
                type_list = np.append(type_list, self.data_types[column_name])
            else:
                print("ERROR with static data - data type for mcdm_old")
                sys.exit(-1)
        return type_list

    def get_data_weights(self):
        data_weights_list = np.array([])
        for key, value in self.data_weights.items():
            data_weights_list = np.append(data_weights_list, value)
        return data_weights_list

    def get_normalized_data(self):
        data_matrix = self.get_data_matrix()
        return minmax_normalization(data_matrix)

    def get_all_models_names(self):
        all_models_names = []
        for index, row in self.evaluated_models_dataframe.iterrows():
            model_name = row[self.results_model_name_column]  # Accessing the first column by name
            all_models_names.append(model_name)
        return all_models_names

    def draw_figure(self, score_list, pymcdm_method_name):
        x_labels = []
        y_labels = []
        for index, row in self.evaluated_models_dataframe.iterrows():
            model_name = row[self.results_model_name_column]  # Accessing the first column by name
            x_labels.append(model_name)
            y_labels.append(score_list[index])

        plt.figure(figsize=(8, 6), num=pymcdm_method_name + self.picture_title)

        plt.bar(x_labels, y_labels)
        plt.title(pymcdm_method_name + self.picture_title, fontsize=14)
        plt.xlabel('Model name', fontsize=8)
        y_label = f'{pymcdm_method_name} values'
        plt.ylabel(y_label, fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')

        file_name = pymcdm_method_name + self.file_postfix + ".png"
        save_path = os.path.join(self.evaluated_models_results_directory, file_name)
        plt.savefig(save_path, dpi=300, format='png')
        plt.close()

    def get_picture_title_for_picture(self, is_confusion_matrix):
        if is_confusion_matrix:
            return " confusion matrix"
        else:
            return " scores and metrics"

    def get_file_postifx_name_for_picture(self, is_confusion_matrix):
        if is_confusion_matrix:
            return "_confusion_matrix_metrics"
        else:
            return "_scores_metrics"
