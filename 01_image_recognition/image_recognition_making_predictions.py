from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]

# Load the json file containing the model structure
f = Path("model_structure.json")
model_structure = f.read_text()
# Recreate the Keras model from json data
model = model_from_json(model_structure)
# Re-load the model's trained weights
model.load_weights("model_weights.h5")

###
# TODO: how do I pass multiple images at the same time?
image_files = ("cat.png", "frog.png", "car.png", "cat01.jpg", "dog01.png", "bird01.jpg", "bird02.jpg", "truck01.jpg")

for image_name in image_files:

    img = image.load_img(image_name, target_size=(32, 32))
    imgage_to_test =  image.img_to_array(img)
    list_of_images = np.expand_dims(imgage_to_test, axis=0)

    result = model.predict(list_of_images)
    single_result = result[0]
    most_likely_class_index = int(np.argmax(single_result))
    class_likelihood = single_result[most_likely_class_index]
    class_label = class_labels[most_likely_class_index]

    print(f'The image {image_name} is a {class_label} - Likelihood {class_likelihood:2f}')


"""
# Load image file to test, resizing to 32x32 pixels (per model requirements)
# image_name = "cat.png"
# image_name = "frog.png"
# image_name = "car.png"
# image_name = "cat01.jpg"
# image_name = "dog01.png"
img = image.load_img(image_name, target_size=(32, 32))
# Convert image to a numpy array
imgage_to_test =  image.img_to_array(img)
# Add fourth dimention (batch - keras expects list of images, not a single one)
list_of_images = np.expand_dims(imgage_to_test, axis=0) # ('0' - the added dimention is at position 0 (the first one))

# Make prediction using the model
result = model.predict(list_of_images)
# We are testing single image - we need to check only the first result
single_result = result[0]
# We get likelihood for all 10 classes - find one with the highest score
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]
# Get the name of this class
class_label = class_labels[most_likely_class_index]

# Print the result
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))
"""