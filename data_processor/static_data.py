evaluated_models_results_path = "..\\results"
evaluated_model_scores_metrics_path = "..\\results\\all_models_scores_metrics.csv"
evaluated_model_conf_matrix_path = "..\\results\\all_models_confusion_matrix_metrics.csv"

file_path_column_tag = 'filepaths'
file_label_column_tag = 'labels'
images_root_directory = "..\\Satellite_Image_Classification_500\\data"

img_size = (224, 224)

all_metrics_from_confusion_matrices = ['precision', 'recall', 'f1-score']

results_model_name_column = 'Model_name'


scores_and_metrics_types = {
    'Training loss score': -1,
    'Training accuracy score': 1,
    'Validation loss score': -1,
    'Validation accuracy score': 1,
    'Test loss score': -1,
    'Test accuracy score': 1,
    'Accuracy score': 1,
    'Log loss score': -1,
    'ROC AUC metric': 1,
    'Precision-Recall AUC metric': 1,
    'Matthews Correlation Coefficient metric': 1,
    "Cohen's Kappa metric": 1,
    'Mean Absolute Error metric': 1,
    'Mean Squared Error metric': -1,
    'R-squared metric': 1
}

scores_and_metrics_weights = {
    'Training loss score': 0.08,
    'Training accuracy score': 0.06,
    'Validation loss score': 0.07,
    'Validation accuracy score': 0.1,
    'Test loss score': 0.07,
    'Test accuracy score': 0.15,
    'Accuracy score': 0.05,
    'Log loss score': 0.05,
    'ROC AUC metric': 0.12,
    'Precision-Recall AUC metric': 0.09,
    'Matthews Correlation Coefficient metric': 0.08,
    "Cohen's Kappa metric": 0.08,
    'Mean Absolute Error metric': 0.035,
    'Mean Squared Error metric': 0.035,
    'R-squared metric': 0.03
}
higher_value_better = ['Training accuracy score', 'Validation accuracy score', 'Test accuracy score', 'Accuracy score',
                       'ROC AUC metric', 'Precision-Recall AUC metric', 'R-squared metric']
closer_to_one_better = ['Matthews Correlation Coefficient metric', "Cohen's Kappa metric"]
lower_value_better = ['Validation loss score', 'Test loss score', 'Training loss score', 'Log loss score',
                      'Mean Absolute Error metric', 'Mean Squared Error metric']

conf_matrix_types = {
    'cloudy_precision': 1,
    'cloudy_recall': 1,
    'cloudy_f1-score': 1,
    'water_precision': 1,
    'water_recall': 1,
    'water_f1-score': 1,
    'desert_precision': 1,
    'desert_recall': 1,
    'desert_f1-score': 1,
    'green_area_precision': 1,
    'green_area_recall': 1,
    'green_area_f1-score': 1
}

conf_matrix_weights = {
    'cloudy_precision': 0.08333333,
    'cloudy_recall': 0.08333333,
    'cloudy_f0.33333-score': 0.08333333,
    'water_precision': 0.08333333,
    'water_recall': 0.08333333,
    'water_f0.33333-score': 0.08333333,
    'desert_precision': 0.08333333,
    'desert_recall': 0.08333333,
    'desert_f0.33333-score': 0.08333333,
    'green_area_precision': 0.08333333,
    'green_area_recall': 0.08333333,
    'green_area_f0.33333-score': 0.08333337
}

