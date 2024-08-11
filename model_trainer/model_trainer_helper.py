from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.models import Sequential

from keras_tuner.tuners import RandomSearch

from functools import partial

from data_processor import data_preprocessor as dpp, static_data as sd


class ModelTrainerHelper:

    def __init__(self, max_trials_for_random_search, executions_per_trial, epochs_tuner_search, batch_size):


        self.max_trials_for_random_search = max_trials_for_random_search
        self.executions_per_trial = executions_per_trial
        self.epochs_tuner_search = epochs_tuner_search
        self.batch_size = batch_size
        self.all_models_results_directory = sd.evaluated_models_results_path

        self.data_preprocessor = dpp.DataPreProcessor()
        self.class_count = self.data_preprocessor.class_count
        self.y_train_converted = self.data_preprocessor.get_converted_y_train()
        self.y_valid_converted = self.data_preprocessor.get_converted_y_valid()
        self.y_test_converted = self.data_preprocessor.get_converted_y_test()
        self.X_train = self.data_preprocessor.X_train
        self.X_valid = self.data_preprocessor.X_valid
        self.image_size = self.data_preprocessor.image_size

    def train_model(self, **model_params):
        sub_directory_to_save_results = f'{list(model_params.items())[0][1]}'
        build_model_partial = partial(self.build_model, **model_params)

        tuner = RandomSearch(
            build_model_partial,
            objective='val_accuracy',
            max_trials=self.max_trials_for_random_search,
            executions_per_trial=self.executions_per_trial,
            directory=self.all_models_results_directory,
            project_name=sub_directory_to_save_results
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        tuner.search(self.X_train, self.y_train_converted, epochs=self.epochs_tuner_search,
                     batch_size=self.batch_size,
                     validation_data=(self.X_valid, self.y_valid_converted), callbacks=[early_stopping])
        best_model = tuner.get_best_models(num_models=1)[0]
        model_file_path = self.data_preprocessor.get_directory_to_save_model_results(self.all_models_results_directory,
                                                                                     sub_directory_to_save_results)
        best_model.save(model_file_path)
        return best_model

    def build_model(self, hp, **model_params):
        base_model_app = list(model_params.items())[1][1]
        # Define base model
        base_model = self.get_base_model(base_model_app)

        for layer in base_model.layers:
            layer.trainable = False

        # Add additional layers for fine-tuning
        model = Sequential([
            base_model,
            Dense(units=hp.Int('dense_units_1', min_value=32, max_value=512, step=32),
                  activation=hp.Choice('activation_1', ['relu', 'tanh', 'sigmoid', 'linear'])),
            Dropout(rate=hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)),
            Dense(units=hp.Int('dense_units_2', min_value=32, max_value=512, step=32),
                  activation=hp.Choice('activation_2', ['relu', 'tanh', 'sigmoid', 'linear'])),
            Dropout(rate=hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)),
            Dense(self.class_count, activation=hp.Choice('output_activation', ['softmax', 'sigmoid', 'linear']))
        ])

        optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad'])
        learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        optimizer = self.get_optimizer(optimizer_choice, learning_rate)

        loss_function = hp.Choice('loss_function', ['cce', 'mse', 'mae', 'bce'])
        loss = self.get_loss_function(loss_function)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model

    def get_base_model(self, base_model_application):
        img_shape = (self.img_size[0], self.img_size[1], 3)
        return base_model_application(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    def get_optimizer(self, optimizer_choice, learning_rate):
        if optimizer_choice == 'adam':
            return Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            return SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            return RMSprop(learning_rate=learning_rate)
        else:
            return Adagrad(learning_rate=learning_rate)

    def get_loss_function(self, loss_function):
        if loss_function == 'cce':
            return 'categorical_crossentropy'
        elif loss_function == 'mse':
            return 'mean_squared_error'
        elif loss_function == 'mae':
            return 'mean_absolute_error'
        else:
            return 'binary_crossentropy'
