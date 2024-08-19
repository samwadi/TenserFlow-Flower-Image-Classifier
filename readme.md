# Flower Image Classifier

This project is part of the Udacity Nanodegree Program: AI with Python and TensorFlow.

## Project Overview

This repository contains a flower image classifier that predicts the type of flower based on an image using a pre-trained model. The model was trained using TensorFlow and the Oxford Flowers 102 dataset.

## Repository Structure

- **`predict.py`**: A command-line tool to predict the top K most likely flower classes for a given image using a trained model.
- **`assets/`**: Contains sample images for testing.
- **`label_map.json`**: JSON file mapping class labels to flower names.
- **`flower_classification_cnn.h5`**: The saved trained model.
- **`Flower_Classifier.ipynb`**: The Jupyter notebook used to create and train the model.
- **`class_labels.json`**: A JSON file that maps class labels to flower names.

## How to Use

### Predicting Flower Classes
flower_classification_cnn.h5 model build was too larg to upload, must run and save the model on your machine!!!
You can use the `predict.py` script to classify a flower image. The script takes several command-line arguments:

```bash
python predict.py /path/to/image --top_k 5 --category_names /path/to/class_labels.json


/path/to/image: Path to the image file you want to classify.
--top_k: (Optional) The number of top classes to return. Default is 1.
--category_names: (Optional) Path to a JSON file mapping class labels to flower names.



Example

python predict.py assets/wild_pansy.jpg --top_k 5 --category_names assets/class_labels.json

This command will return the top 5 most likely flower types for the image wild_pansy.jpg.


Notebook Overview
The Jupyter notebook Flower_Classifier.ipynb contains the process used to create and train the model, including data preprocessing, model architecture, training, and evaluation.

```
