import pandas as pd
import single_model_evaluator_helper as smeh
import model_evaluator_helper as meh
from data_processor import static_data as sd


class ModelEvaluator:
    def __init__(self, model_evaluator_helper):
        self.model_evaluator_helper = model_evaluator_helper

        self.evaluated_model_scores_metrics_path = sd.evaluated_model_scores_metrics_path
        self.evaluated_model_conf_matrix_path = sd.evaluated_model_conf_matrix_path

    def evaluate_saved_model(self):
        all_trained_models_dict = self.model_evaluator_helper.find_saved_models_in_result_path()
        all_models_scores_and_metrics = []
        all_models_confusion_matrix_metrics = []
        for model_name, model_path in all_trained_models_dict.items():
            single_model_evaluator_helper = smeh.SingleModelEvaluatorHelper(model_path, model_name)
            single_model_evaluator_helper.save_denormalized_confusion_matrix_plot()
            all_models_scores_and_metrics.append(single_model_evaluator_helper.calculate_scores_and_metrics())
            all_models_confusion_matrix_metrics.append(single_model_evaluator_helper.calculate_confusion_matrix_metrics())
        self.save_scores_and_metrics_to_csv(all_models_confusion_matrix_metrics, all_models_scores_and_metrics)

    def save_scores_and_metrics_to_csv(self, confusion_matrix_metrics, scores_and_metrics):

        all_trained_models_names = self.model_evaluator_helper.get_all_trained_model_names()
        normalized_metrics_from_confusion_matrix_df = self.convert_normalized_metrics_from_confusion_matrix_to_df(
            confusion_matrix_metrics, all_trained_models_names)
        normalized_metrics_from_confusion_matrix_df.to_csv(self.evaluated_model_conf_matrix_path, index=False)
        scores_and_metrics = self.model_evaluator_helper.round_model_for_score_and_metrics(scores_and_metrics, 3)
        scores_and_metrics_df = pd.DataFrame(scores_and_metrics)
        scores_and_metrics_df.to_csv(self.evaluated_model_scores_metrics_path, index=False)



    def convert_normalized_metrics_from_confusion_matrix_to_df(self, confusion_matrix_metrics,
                                                               all_trained_models_names):
        normalized_metrics_from_confusion_matrix = self.normalize_metrics(confusion_matrix_metrics,
                                                                          all_trained_models_names)
        return pd.DataFrame(normalized_metrics_from_confusion_matrix[1:],
                            columns=normalized_metrics_from_confusion_matrix[0])

    def normalize_metrics(self, input_data, all_model_name_list):
        column_header_names = self.model_evaluator_helper.get_column_header_names()
        normalized_confusion_matrix_content = []
        for idx, model_data in enumerate(input_data):
            model_name = all_model_name_list[idx]
            model_metrics = self.model_evaluator_helper.process_model_data(model_data)
            normalized_confusion_matrix_content.append([model_name] + model_metrics)
        formatted_output = [column_header_names] + normalized_confusion_matrix_content
        return formatted_output


def main():
    model_evaluator_helper = meh.ModelEvaluatorHelper()
    model_evaluator = ModelEvaluator(model_evaluator_helper)
    model_evaluator.evaluate_saved_model()
    model_evaluator_helper.plot_scores_and_metrics_for_all_models()


if __name__ == "__main__":
    main()
