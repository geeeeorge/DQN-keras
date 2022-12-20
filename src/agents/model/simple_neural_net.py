from keras.layers import Input, concatenate, Dense, Flatten
from keras.models import Model


class SimpleNeuralNet:
    def __init__(self, input_shape, nb_output):
        self.input_shape = input_shape
        self.nb_output = nb_output

    def _build_network(self):
        input_layer = Input(shape=self.input_shape)
        x = Flatten()(input_layer)
        x = Dense(32, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        output_layer = Dense(self.nb_output, activation='linear')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.summary()

        return model

    def model(self):
        return self._build_network()
