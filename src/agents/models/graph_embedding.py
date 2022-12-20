from keras.layers import Input, concatenate, Dense, Flatten
from keras.models import Model


class GraphEmbedding:

def build_q_network(input_shape, nb_output):
    input_layer = Input(shape=input_shape)
    x = Flatten()(input_layer)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(nb_output, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
