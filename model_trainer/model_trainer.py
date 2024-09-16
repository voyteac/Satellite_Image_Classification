from keras._tf_keras.keras.applications import (EfficientNetB3, ResNet50, VGG16, VGG19, Xception, MobileNet,
                                                DenseNet201, EfficientNetV2B3)


import keras
import tensorflow
import model_trainer_helper as mth


class ModelTrainer:
    def __init__(self, model_trainer_helper, base_model_list):
        self.model_trainer_helper = model_trainer_helper
        self.base_model_list = base_model_list

    def train_models(self):
        for model_name, base_model_application in self.base_model_list.items():
            self.model_trainer_helper.train_model(model_name=model_name, base_model=base_model_application)


def main():
    epochs_tuner_search = 10
    max_trials_for_random_search = 10
    executions_per_trial = 1
    batch_size = 16
    base_model_list = {
        'DenseNet201': DenseNet201,
        'EfficientNetB3': EfficientNetB3,
        'EfficientNetV2B3': EfficientNetV2B3,
        'MobileNet': MobileNet,
        'ResNet50': ResNet50,
        'VGG16': VGG16,
        'VGG19': VGG19,
        'Xception': Xception
    }

    model_trainer_helper = mth.ModelTrainerHelper(max_trials_for_random_search, executions_per_trial,
                                                  epochs_tuner_search, batch_size)
    model_trainer = ModelTrainer(model_trainer_helper, base_model_list)
    model_trainer.train_models()


if __name__ == "__main__":
    main()