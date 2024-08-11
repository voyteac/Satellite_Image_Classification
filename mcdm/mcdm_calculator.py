import numpy as np

from pymcdm.methods import TOPSIS, VIKOR, SPOTIS, PROMETHEE_II

from data_processor import static_data as sd
import mcdm_calculator_helper as mch


class McdmCalculator:
    def __init__(self, mcdm_calculator_helper):
        self.mcdm_calculator_helper = mcdm_calculator_helper
        self.load_all_models_df_only_data = self.mcdm_calculator_helper.drop_model_name_from_df_data()

    def calculate_rounded_scores_all_methods(self):
        data_types = self.mcdm_calculator_helper.get_data_types()
        data_weights = self.mcdm_calculator_helper.get_data_weights()
        normalized_data = self.mcdm_calculator_helper.get_normalized_data()
        model_names = self.mcdm_calculator_helper.get_all_models_names()

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



        # Print rounded scores
        print("\nScores (Rounded):")
        header = "Method".ljust(15) + "\t" + "\t".join(model_names)
        print(header)
        for method, values in rounded_scores_all_methods.items():
            formatted_values = "\t\t\t".join(map(str, values))
            print(f"{method.ljust(15)}\t{formatted_values}")

        # Print rankings
        print("\nRankings:")
        print(header)  # Reuse the header from above
        for method, values in rankings_all_methods.items():
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
    mcdm_calculator.calculate_rounded_scores_all_methods()

    ### confusion matrix
    evaluated_model_conf_matrix_path = sd.evaluated_model_conf_matrix_path
    data_types = sd.conf_matrix_types
    data_weights = sd.conf_matrix_weights
    mcdm_calculator_helper_cm = mch.McdmCalculatorHelper(evaluated_model_conf_matrix_path, data_types,
                                                         data_weights, True)

    mcdm_calculator = McdmCalculator(mcdm_calculator_helper_cm)
    mcdm_calculator.calculate_rounded_scores_all_methods()


if __name__ == "__main__":
    main()