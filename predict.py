import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
import logging

# Hide unnecessary warnings
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

IMAGE_SIZE = 224


def load_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    return loaded_model


def process_image(numpy_image):
    """
    In this method, we will process image by resizing it to a given size and normalizing it.
    After that, we will also convert image to numpy array to use it in the model prediction.
    :param numpy_image: path for the desired flower to predict classification of this flower
    :return: returns processed image as numpy array
    """
    processed_image = tf.convert_to_tensor(numpy_image, dtype=tf.float32)
    processed_image = tf.image.resize(processed_image, (IMAGE_SIZE, IMAGE_SIZE))
    processed_image /= 255
    return processed_image.numpy()


def process_value_array(top_k_prediction_values):
    return tf.squeeze(top_k_prediction_values).numpy()


def increment_label_by_one(label):
    return label + 1


def decode_text_for_array(array_of_labels):
    for index, value in enumerate(array_of_labels):
        array_of_labels[index] = value.decode('utf-8')
    return array_of_labels


def process_label_text_array(top_k_prediction_indices):
    top_k_prediction_indices_incremented_by_one = tf.map_fn(increment_label_by_one, top_k_prediction_indices)
    top_k_prediction_indices_as_string = tf.strings.as_string(top_k_prediction_indices_incremented_by_one)
    top_k_prediction_indices_squeezed = tf.squeeze(top_k_prediction_indices_as_string)
    top_k_prediction_indices_as_array = top_k_prediction_indices_squeezed.numpy()
    top_k_prediction_indices_decoded_text = decode_text_for_array(top_k_prediction_indices_as_array)
    return top_k_prediction_indices_decoded_text


def predict(image_path, model, top_k=5):
    given_test_image = Image.open(image_path)
    numpy_test_image = np.asarray(given_test_image)
    processed_test_image = process_image(numpy_test_image)
    expanded_test_image = np.expand_dims(processed_test_image, axis=0)

    prediction = model.predict(expanded_test_image)

    top_k_predictions = tf.math.top_k(prediction, k=top_k, sorted=True)

    top_k_prediction_values = process_value_array(top_k_predictions.values)
    top_k_prediction_indices = process_label_text_array(top_k_predictions.indices)

    return top_k_prediction_values, top_k_prediction_indices


def load_class_names(category_names_path):
    with open(category_names_path, 'r') as f:
        return json.load(f)


def retrieve_top_k_class_names(top_k_prediction_indices, dictionary_of_class_names):
    top_k_class_names = []

    for prediction_label in top_k_prediction_indices:
        top_k_class_names.append(dictionary_of_class_names[prediction_label])

    return top_k_class_names


def print_top_k_prediction_details(top_k_prediction_values, list_of_top_k_class_names):
    for i, prediction_value in enumerate(top_k_prediction_values):
        print('-----[ {} ]-----'.format((i+1)))
        print('Class name: ', list_of_top_k_class_names[i])
        print('Confident rate: {:.2%}'.format(prediction_value))
        print('\n')



if __name__ == '__main__':
    # Define arg parser
    parser = argparse.ArgumentParser(description='Predict a flower type')

    parser.add_argument('--input', action='store', dest='input', default='./test_images/orange_dahlia.jpg')
    parser.add_argument('--model', action='store', dest='model', default='./models/1650973761.h5')
    parser.add_argument('--top_k', action='store', dest='top_k', default=5, type=int)
    parser.add_argument('--category_names', action='store', dest="category_names", default='./label_map.json')

    args = parser.parse_args()

    input_image_path = args.input
    model_path = args.model
    top_k = args.top_k
    category_names_path = args.category_names

    model = load_model(model_path)

    top_k_prediction_values, top_k_prediction_indices = predict(input_image_path, model, top_k)

    class_names = load_class_names(category_names_path)

    list_of_top_k_class_names = retrieve_top_k_class_names(top_k_prediction_indices, class_names)

    print_top_k_prediction_details(top_k_prediction_values, list_of_top_k_class_names)