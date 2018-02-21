# DeepModels
Implementations of various deep learning models using Keras.

## Requirements
An existing installation of either Tensorflow or Theano.

## Installation
```
pip install deep-models
```

## Usage

The models are implemented using Keras and instantiation returns a Keras Model object unless otherwise noted.

### Wide Residual Network
```python3
from deep_models import wide_residual_network as wrn

# Load your data
trainX = ...
trainY = ...
img_shape = (32, 32, 3)

# Create the model
# k is the width, 6 * n + 4 is the depth
model = wrn.build_model(img_shape, classes=10, n=4, k=10, dropout=0.3)

# Train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(
    trainX, trainY,
    batch_size=128,
    epochs=100,
    validation_split=0.2)
```

## Examples
Some working examples are available in the notebooks directory.

## License
See LICENSE file
