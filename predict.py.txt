import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)

    predictions = model.predict(processed_image)
    probs, classes = tf.math.top_k(predictions, k=top_k)
    return probs.numpy()[0], classes.numpy()[0]

def load_class_names(category_names):
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Flower image classifier')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')

    args = parser.parse_args()

    model = load_model('flower_classification_cnn.h5')

    probs, classes = predict(args.image_path, model, args.top_k)

    if args.category_names:
        class_names = load_class_names(args.category_names)
        classes = [class_names[str(cls)] for cls in classes]

    print("Probabilities:", probs)
    print("Classes:", classes)

if __name__ == '__main__':
    main()
