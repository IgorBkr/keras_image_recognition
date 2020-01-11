from pathlib import Path
import numpy as np 
import joblib
from keras.preprocessing import image
from keras.applications import vgg16

# Path to training data
dog_path = Path("training_data") / "dogs"
not_dog_path = Path("training_data") / "not_dogs"

images = []
labels = []

# Load non-dog images
for img in not_dog_path.glob("*.png"):
    img = image.load_img(img)
    image_array = image.img_to_array(img) # convert image to numpy array
    images.append(image_array)  # append to list of images
    labels.append(0) # label '0' for not dogs
# Load dog images
for img in dog_path.glob("*.png"):
    img = image.load_img(img)
    image_array = image.img_to_array(img)
    images.append(image_array)
    labels.append(1)

# Create a numpy arrays with all the images and labels (keras expected input)
x_train = np.array(images)
y_train = np.array(labels)
# Normalize image data
x_train = vgg16.preprocess_input(x_train)

# Load pre-trained network (to use as a feture extractor): one which is trained on imagenet dataset; 
# with no last layer (we need no predictions - just feature extraction); 
# using small 64x64 images just for speed - for better result use something like 224x224...
pretrained_nn = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(64, 64, 3)) 

# Extract features for each image
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")
# Save matching expected values
joblib.dump(y_train, "y_train.dat")