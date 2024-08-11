The satellite image classification using the Keras framework.

The project aims to create and train convolutional neural network (CNN) models for image classification tasks. Specifically, it seeks to:
1.	Automate Model Development
The script automates the process of building, tuning, and training CNN models by providing functions for data preprocessing, model architecture definition, hyperparameter tuning, and model training.
2.	Optimise Performance
By employing techniques like Random Search for hyperparameter optimisation and Early Stopping to prevent overfitting, the script aims to find the best-performing model configuration for the given dataset.
3.	Achieve High Accuracy
The ultimate goal is to develop models that achieve high accuracy in classifying images into their respective categories or classes. This is crucial for various applications such as object recognition, medical imaging, satellite image analysis, and more.
4.	Provide Flexibility
The script allows flexibility in choosing different pre-trained CNN architectures and tuning hyperparameters to adapt to different datasets and tasks.
Overall, the goal is to streamline the process of creating effective image classification models, making them accessible to users who may not have extensive expertise in machine learning or deep learning.


Dataset
The dataset chosen for the Project originates from the Kaggle platform. The dataset “Satellite Image Classification”*contains 4 different classes of satellite pictures:
•	cloudy,
•	desert,
•	green area,
•	water.
Each group of pictures consists of 1500 pictures. Due to performance issues, the dataset has been limited to 500 photographs. The selection has been done randomly.
*) https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
