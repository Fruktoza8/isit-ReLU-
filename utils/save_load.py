import os
import numpy as np
import json


def save_weights_json(network, filepath="weights.json"):
    data = {
        'weights': [weight.tolist() for weight in network.weights],  # Convert weights to lists
        'biases': [bias.tolist() for bias in network.biases]         # Convert biases to lists
    }
    with open(filepath, 'w') as f:
        json.dump(data, f)


def load_weights_json(network, file_path="weights/saved_weights.json"):
    # Чтение данных из файла JSON
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Восстанавливаем веса и байасы
    network.layers = [np.array(layer) for layer in data['layers']]
    network.biases = [np.array(bias) for bias in data['biases']]
