from keras.layers import Input, BatchNormalization, Conv2D, Activation, Dense
from keras.layers import Dropout, Flatten, Add, AveragePooling2D
from keras.models import Model
from keras.regularizers import l2
from keras import backend

_CHANNEL_AXIS = 1 if backend.image_data_format() == "channels_first" else -1


class _ConvStack:
    def __init__(self, base_width=16, n=2, k=8, dropout=0.0, strides=(1, 1),
                 weight_decay=0.0005):
        self.base_width = base_width
        self.n = n
        self.k = k
        self.dropout = dropout
        self.strides = strides
        self.weight_decay = weight_decay

    def __call__(self, x):
        # Shortcut layer
        x = BatchNormalization(axis=_CHANNEL_AXIS,
                               momentum=0.1,
                               epsilon=1e-5,
                               gamma_initializer='uniform',
                               gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(x)
        x = Activation('relu')(x)

        # Branch off x
        z = Conv2D(self.base_width * self.k, (3, 3),
                   strides=self.strides,
                   padding='same',
                   use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay))(x)
        z = Dropout(self.dropout)(z)
        z = BatchNormalization(axis=_CHANNEL_AXIS,
                               momentum=0.1,
                               epsilon=1e-5,
                               gamma_initializer='uniform',
                               gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay))(z)
        z = Activation('relu')(z)
        z = Conv2D(self.base_width * self.k, (3, 3),
                   strides=(1, 1),
                   padding='same',
                   use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay))(z)

        # Second branch
        x = Conv2D(self.base_width * self.k, (1, 1),
                   strides=self.strides,
                   padding='same',
                   use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay))(x)

        # Merge branches
        x = Add()([z, x])

        # Residual layers
        for i in range(self.n - 1):
            # Branch off x
            z = BatchNormalization(axis=_CHANNEL_AXIS,
                                   momentum=0.1,
                                   epsilon=1e-5,
                                   gamma_initializer='uniform',
                                   gamma_regularizer=l2(self.weight_decay),
                                   beta_regularizer=l2(self.weight_decay))(x)
            z = Activation('relu')(z)
            z = Conv2D(self.base_width * self.k, (3, 3),
                       strides=(1, 1),
                       padding='same',
                       use_bias=False,
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.weight_decay))(z)
            z = Dropout(self.dropout)(z)
            z = BatchNormalization(axis=_CHANNEL_AXIS,
                                   momentum=0.1,
                                   epsilon=1e-5,
                                   gamma_initializer='uniform',
                                   gamma_regularizer=l2(self.weight_decay),
                                   beta_regularizer=l2(self.weight_decay))(z)
            z = Activation('relu')(z)
            z = Conv2D(self.base_width * self.k, (3, 3),
                       strides=(1, 1),
                       padding='same',
                       use_bias=False,
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.weight_decay))(z)

            # Merge branches
            x = Add()([z, x])

        return x


def build_model(input_shape, classes=10, n=2, k=8, dropout=0.0,
                weight_decay=0.0005, verbose=False):
    # Initial layers
    _input = Input(input_shape)
    x = Conv2D(16, (3, 3), strides=(1, 1),
               padding='same',
               use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(_input)

    # Residual layers
    x = _ConvStack(16, n=n, k=k, dropout=dropout, strides=(1, 1))(x)
    x = _ConvStack(32, n=n, k=k, dropout=dropout, strides=(2, 2))(x)
    x = _ConvStack(64, n=n, k=k, dropout=dropout, strides=(2, 2))(x)

    # Output layers
    x = BatchNormalization(axis=_CHANNEL_AXIS,
                           momentum=0.1,
                           epsilon=1e-5,
                           gamma_initializer='uniform',
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8, 8), (1, 1), padding='same')(x)
    x = Flatten()(x)
    _output = Dense(classes,
                    activation='softmax',
                    kernel_regularizer=l2(weight_decay),
                    bias_regularizer=l2(weight_decay))(x)

    if verbose:
        print('WRN-{}-{}{} created'.format(6 * n + 4, k,
                                           '-dropout' if dropout > 0 else ''))

    return Model(inputs=_input, outputs=_output)
