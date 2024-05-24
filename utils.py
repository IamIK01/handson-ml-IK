import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def set_seed(seed: int=42)-> None:
    tf.random.set_seed(42)
    np.random.seed(42)

def show_random_images(n: int,  x: np.ndarray, y: np.ndarray, a=None, b=None, figsize=(15, 15)):
    indices = np.random.choice(np.arange(x.shape[0]), n, replace=False)
    images = x[indices]
    labels = y[indices]

    if np.max(images) < 254.0:
        images = np.uint8(x*255)

    if a is None or b is None:
        # Calculate a and b based on n
        a = int(np.ceil(np.sqrt(n)))
        b = int(np.ceil(n / a))
    
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(a, b, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[np.argmax(labels[i])])
    plt.show()

# Function to create directory for storing models and training data
def get_chapter_directory(chapter_num):
    dir_name = f'chapter_{chapter_num}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name
