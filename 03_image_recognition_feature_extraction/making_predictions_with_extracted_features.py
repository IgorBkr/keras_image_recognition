from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np 
from keras.applications import vgg16

# Recreate model from stored structure and weights
f = Path("feature_model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("feature_model_weights.h5")

# Load image to test (model was trained on 64x64 images)
img_name = "dog.png"
# img_name = "not_dog.png"
img = image.load_img(img_name, target_size=(64, 64))
image_array = image.img_to_array(img)
images = np.expand_dims(image_array, axis=0)
# Normalize the data
images = vgg16.preprocess_input(images)

# Use pre-trained network to extract features (same as we did in training)
feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
features = feature_extraction_model.predict(images)

# Make predictions with our model, using extracted features
results = model.predict(features)

# (We are testing single image with one possible class - check only first element of first result)
single_result = results[0][0]

# Print
print("Image {} contains a dog with likelihood of: {}%".format(img_name, int(single_result*100)))
