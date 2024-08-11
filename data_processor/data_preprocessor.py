import numpy as np
import pandas as pd
import os
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.applications.efficientnet import preprocess_input
from keras._tf_keras.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import static_data as sd


class DataPreProcessor:
    def __init__(self):
        self.image_size = sd.img_size
        self.file_path_column_tag = sd.file_path_column_tag
        self.file_label_column_tag = sd.file_label_column_tag
        self.images_root_directory = sd.images_root_directory
        self.label_encoder = LabelEncoder()
        image_array, label_array = self.preprocess_images(self.image_size)

        self.X_train, self.X_test_valid, self.y_train, self.y_test_valid = (
            train_test_split(image_array, label_array, test_size=0.2, random_state=42))
        self.X_valid, self.X_test, self.y_valid, self.y_test =\
            train_test_split(self.X_test_valid, self.y_test_valid, test_size=0.5, random_state=42)

        self.class_count = self.get_class_count()

    def get_class_count(self):
        return len(np.unique(self.y_train))

    def get_encoded_y_train(self):
        return self.label_encoder.fit_transform(self.y_train)
    def get_encoded_y_valid(self):
        return self.label_encoder.transform(self.y_valid)

    def get_encoded_y_test(self):
        return self.label_encoder.transform(self.y_test)

    def get_converted_y_train(self):
        return to_categorical(self.get_encoded_y_train(), num_classes=self.class_count)

    def get_converted_y_valid(self):
        return to_categorical(self.get_encoded_y_valid(), num_classes=self.class_count)

    def get_converted_y_test(self):
        return to_categorical(self.get_encoded_y_test(), num_classes=self.class_count)

    def get_directory_to_save_model_results(self, root_directory_for_results, sub_directory_to_save_results):
        file_name = sub_directory_to_save_results + '.keras'
        print(f'File name: {file_name}')
        return os.path.join(root_directory_for_results, sub_directory_to_save_results, file_name)

    def get_file_list_with_labels_df(self):
        file_path_series = pd.Series(self.get_file_path_list(), name=self.file_path_column_tag)
        file_label_series = pd.Series(self.get_file_label_list(), name=self.file_label_column_tag)
        return pd.concat([file_path_series, file_label_series], axis=1)

    def get_file_label_list(self):
        file_label_list = []
        label_list = os.listdir(self.images_root_directory)
        for label_list in label_list:
            file_list = os.listdir(os.path.join(self.images_root_directory, label_list))
            for file in file_list:
                file_label_list.append(label_list)
        return file_label_list

    def get_file_path_list(self):
        file_path_list = []
        directory_list = os.listdir(self.images_root_directory)
        for directory in directory_list:
            directory_path = os.path.join(self.images_root_directory, directory)
            file_list = os.listdir(directory_path)
            for file in file_list:
                file_path_list.append(os.path.join(directory_path, file))
        return file_path_list

    def preprocess_images(self, img_size):
        df = self.get_file_list_with_labels_df()
        filepaths = df[self.file_path_column_tag]
        labels = df[self.file_label_column_tag]
        images = []
        for filepath in filepaths:
            img = load_img(filepath, target_size=img_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
        X = np.array(images)
        y = np.array(labels)
        return X, y


