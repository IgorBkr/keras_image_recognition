from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import joblib

# Load data sets (which are extracted features in this case, not raw images)
x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

# Create model and layers (only Dense layers will be re-trained; all Conv. layers were used form vgg16)
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=10,
    shuffle=True
)

# Save network structure
model_structure = model.to_json()
f = Path("feature_model_structure.json")
f.write_text(model_structure)
# Save weights
model.save_weights("feature_model_weights.h5")