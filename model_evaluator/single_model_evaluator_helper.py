import os

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from keras._tf_keras.keras.models import load_model
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef,
                             cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score, log_loss)
from sklearn.metrics import confusion_matrix, classification_report

from data_processor import data_preprocessor as dpp, static_data as sd


class SingleModelEvaluatorHelper:
    def __init__(self, loaded_model_path, model_name):
        self.loaded_model_path = loaded_model_path
        self.loaded_model_name = model_name
        self.data_preprocessor = dpp.DataPreProcessor()
        self.loaded_model = load_model(loaded_model_path)

        self.all_models_results_directory = sd.evaluated_models_results_path
        self.results_model_name_column = sd.results_model_name_column
        self.evaluated_model_scores_metrics_path = sd.evaluated_model_scores_metrics_path

        self.images_root_directory = self.data_preprocessor.images_root_directory
        self.X_test = self.data_preprocessor.X_test
        self.X_train = self.data_preprocessor.X_train
        self.X_valid = self.data_preprocessor.X_valid
        self.y_train_converted = self.data_preprocessor.get_converted_y_train()
        self.y_valid_converted = self.data_preprocessor.get_converted_y_valid()
        self.y_test_converted = self.data_preprocessor.get_converted_y_test()
        self.y_test_encoded = self.data_preprocessor.get_encoded_y_test()
        self.y_train_encoded = self.data_preprocessor.get_encoded_y_train()

        self.x_predictions = self.loaded_model.predict(self.X_test)
        self.y_predictions = np.argmax(self.x_predictions, axis=1)
        self.train_score = self.loaded_model.evaluate(self.X_train, self.y_train_converted, verbose=1)
        self.validation_score = self.loaded_model.evaluate(self.X_valid, self.y_valid_converted, verbose=1)
        self.test_scores = self.loaded_model.evaluate(self.X_test, self.y_test_converted, verbose=1)

    def calculate_scores_and_metrics(self):
        return {
            self.results_model_name_column: self.loaded_model_name,
            'Training loss score': self.train_score[0],
            'Training accuracy score': self.train_score[1],
            'Validation loss score': self.validation_score[0],
            'Validation accuracy score': self.validation_score[1],
            'Test loss score': self.test_scores[0],
            'Test accuracy score': self.test_scores[1],
            'Accuracy score': accuracy_score(self.y_test_encoded, self.y_predictions),
            'Log loss score': log_loss(self.y_test_encoded, self.x_predictions),
            'ROC AUC metric': roc_auc_score(self.y_test_converted, self.x_predictions, average='macro',
                                            multi_class='ovo'),
            'Precision-Recall AUC metric': average_precision_score(self.y_test_converted, self.x_predictions,
                                                                   average='macro'),
            'Matthews Correlation Coefficient metric': matthews_corrcoef(self.y_test_encoded, self.y_predictions),
            "Cohen's Kappa metric": cohen_kappa_score(self.y_test_encoded, self.y_predictions),
            'Mean Absolute Error metric': mean_absolute_error(self.y_test_encoded, self.y_predictions),
            'Mean Squared Error metric': mean_squared_error(self.y_test_encoded, self.y_predictions),
            'R-squared metric': r2_score(self.y_test_encoded, self.y_predictions)
        }

    def get_all_labels_based_on_image_directories(self):
        return os.listdir(self.images_root_directory)

    def calculate_confusion_matrix_metrics(self):
        return classification_report(self.y_test_encoded, self.y_predictions, target_names=self.get_class_name_list(),
                                     output_dict=True)

    def save_denormalized_confusion_matrix_plot(self):

        cm = confusion_matrix(self.y_test_encoded, self.y_predictions)
        title = f'Confusion Matrix for model: "{self.loaded_model_name}".'
        plt.figure(figsize=(14, 10), num=self.loaded_model_name)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title, fontsize=16)
        plt.colorbar()

        class_name_list = self.get_class_name_list()
        tick_marks = np.arange(len(class_name_list))
        plt.xticks(tick_marks, class_name_list, fontsize=10)
        plt.yticks(tick_marks, class_name_list, fontsize=10)

        thresh = np.max(cm) / 2.
        for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(column, row, format(cm[row, column], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[row, column] > thresh else 'black', fontdict=
                     {'family': 'serif', 'size': 12, 'weight': 'normal'})

        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        save_path = self.all_models_results_directory + "\\" + self.loaded_model_name + ".png"
        plt.savefig(save_path, dpi=300, format='png')
        plt.close()

    def get_class_name_list(self):
        return os.listdir(self.images_root_directory)

    def save_scores_and_metrics_plot(self):
        higher_value_better = sd.higher_value_better
        closer_to_one_better = sd.closer_to_one_better
        lower_value_better = sd.lower_value_better

        load_all_models_df = pd.read_csv(self.evaluated_model_scores_metrics_path)
        load_all_models_df_only_data = load_all_models_df.drop(load_all_models_df.columns[:2], axis=1)

        start_column = 2
        for score in load_all_models_df_only_data:
            score_name = load_all_models_df.columns[start_column]
            plt.figure(figsize=(10, 6), num=score_name)
            x_labels = load_all_models_df[self.results_model_name_column]
            y_labels = load_all_models_df_only_data[score]
            plt.ylim(min(y_labels) - 0.1 * (max(y_labels) - min(y_labels)),
                     max(y_labels) + 0.1 * (max(y_labels) - min(y_labels)))
            plt.title(score_name)
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
