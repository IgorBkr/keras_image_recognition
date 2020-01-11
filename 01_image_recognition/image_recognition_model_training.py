import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path

# Load data set (returns 4 arrays; x-arrays are images, y-arrays are labels)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data set to 0-to-1 range
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train/255
x_test = x_test/255

# Convert class vectors to binary class matrices
# Our labels are single values from 0 to 9.
# Instead, we want each label to be an array with on element set to 1 and and the rest set to 0.
y_train = keras.utils.to_categorical(y_train, 10) # (cfar10 has 10 categories)
y_test = keras.utils.to_categorical(y_test, 10)

# Create a model and add layers
model = Sequential()

model.add(Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(32, 32, 3))) # (32 filters; 3x3 tiles; padding - for kernel pixels beyond image; input_shape - images size)
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.25)) # (randomly cut off up to %25 traffic - to prevent memorization by network, as opposite to learning)

model.add(Conv2D(64, (3,3), padding="same", activation="relu")) # (layer with 64 filters)
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.25))

model.add(Flatten()) # (to 1D array)
model.add(Dense(512, activation="relu")) # (512 nodes; images are 32x32)
model.add(Dropout(0.5)) # (%50 - to make it work hard)
model.add(Dense(10, activation="softmax")) # (output layer for 10 categories; softmax: all 10 values would add to 1 (%100))

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Print summary of the model
# model.summary()

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=32, # TODO: adjust for memory/speed, if needed
    # epochs=30,
    epochs=50,
    verbose=2, # '2' = 1 line per epoch (default is 1)
    validation_data=(x_test, y_test),
    shuffle=True
)

# Save neural network structure (to be recalled later for making predictions)
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save network's trained weights
model.save_weights("model_weights.h5")
