import numpy as np

from pymcdm.methods import TOPSIS, VIKOR, SPOTIS, PROMETHEE_II

from data_processor import static_data as sd
import mcdm_calculator_helper as mch


class McdmCalculator:
    def __init__(self, mcdm_calculator_helper):
        self.mcdm_calculator_helper = mcdm_calculator_helper
        self.load_all_models_df_only_data = self.mcdm_calculator_helper.drop_model_name_from_df_data()
        self.model_names = self.mcdm_calculator_helper.get_all_models_names()


    def calculate_scores_and_rankings_all_methods(self):
        data_types = self.mcdm_calculator_helper.get_data_types()
        data_weights = self.mcdm_calculator_helper.get_data_weights()
        normalized_data = self.mcdm_calculator_helper.get_normalized_data()

        methods = {
            'VIKOR': VIKOR(),
            'TOPSIS': TOPSIS(),
            'PROMETHEE_II': PROMETHEE_II('usual'),
            'SPOTIS': SPOTIS(self.mcdm_calculator_helper.get_bounds())
        }

        rankings_all_methods = {}
        scores_all_methods = {}
        for name, method_function in methods.items():
            scores_all_methods[name] = method_function(normalized_data, data_weights, data_types)
            rankings_all_methods[name] = method_function.rank(scores_all_methods[name])
            self.mcdm_calculator_helper.draw_figure(scores_all_methods[name], name)
        rounded_scores_all_methods = {method: np.round(values, 3) for method, values in scores_all_methods.items()}
        return rounded_scores_all_methods, rankings_all_methods

    def plot_mcdm_results(self, data_to_be_plotted, plot_scores=True):
        # Print rankings
        if plot_scores:
            print("\nScores (Rounded):")
        else:
            print("\nRankings:")
        header = "Method".ljust(15) + "\t" + "\t".join(self.model_names)
        print(header)  # Reuse the header from above
        for method, values in data_to_be_plotted.items():
            formatted_values = "\t\t\t".join(map(str, values))
            print(f"{method.ljust(15)}\t{formatted_values}")


def main():
    ### metrics and scores
    evaluated_model_scores_metrics_path = sd.evaluated_model_scores_metrics_path
    data_types = sd.scores_and_metrics_types
    data_weights = sd.scores_and_metrics_weights
    mcdm_calculator_helper_metrics_score = mch.McdmCalculatorHelper(evaluated_model_scores_metrics_path, data_types,
                                                                    data_weights, False)
    mcdm_calculator = McdmCalculator(mcdm_calculator_helper_metrics_score)
    mcdm_calculator.calculate_scores_and_rankings_all_methods()

    scores_and_rankings_metrics_and_scores = mcdm_calculator.calculate_scores_and_rankings_all_methods()
    scores_metrics_and_scores = scores_and_rankings_metrics_and_scores[0]
    rankings_metrics_and_scores = scores_and_rankings_metrics_and_scores[1]
    mcdm_calculator.plot_mcdm_results(scores_metrics_and_scores, True)
    mcdm_calculator.plot_mcdm_results(rankings_metrics_and_scores, False)


    ### confusion matrix
    evaluated_model_conf_matrix_path = sd.evaluated_model_conf_matrix_path
    data_types = sd.conf_matrix_types
    data_weights = sd.conf_matrix_weights
    mcdm_calculator_helper_conf_matrix = mch.McdmCalculatorHelper(evaluated_model_conf_matrix_path, data_types,
                                                         data_weights, True)

    mcdm_calculator = McdmCalculator(mcdm_calculator_helper_conf_matrix)
    mcdm_calculator.calculate_scores_and_rankings_all_methods()

    scores_and_rankings_metrics_and_scores = mcdm_calculator.calculate_scores_and_rankings_all_methods()
    scores_conf_matrix = scores_and_rankings_metrics_and_scores[0]
    rankings_conf_matrix = scores_and_rankings_metrics_and_scores[1]

    mcdm_calculator.plot_mcdm_results(scores_conf_matrix, True)
    mcdm_calculator.plot_mcdm_results(rankings_conf_matrix, False)


if __name__ == "__main__":
    main()