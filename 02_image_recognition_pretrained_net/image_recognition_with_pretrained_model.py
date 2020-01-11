import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

# Load Keras VGG16 model (pre-trained against ImageNet database)
model = vgg16.VGG16()

# Load image
img = image.load_img("bay.jpg", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Normalize image (so pixel values are between 0 and 1)
x = vgg16.preprocess_input(x)

# Run predictions
predictions = model.predict(x) # (returns 1K elements array of classes with their likelyhood)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = vgg16.decode_predictions(predictions, top=9) # (get top 9 matches)

# Print results
print('Top predictions for this image:')
for _, name, likelihood in predicted_classes[0]:
    print(f'Prediction: {name} - {likelihood:2f}')