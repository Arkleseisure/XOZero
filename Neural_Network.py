from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import Input, Model
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Activation, Add
from keras.optimizers import SGD
from keras.regularizers import l2
from tensorflow import nn
from tensorflow.compat.v1 import Session, ConfigProto, GPUOptions
from tensorflow.compat.v1.keras.backend import set_session
board_size = 3
weight_decay = 1e-4

tf_config = ConfigProto(gpu_options=GPUOptions(allow_growth=True))
session = Session(config=tf_config)
set_session(session)


def make_net(num_blocks):
    input = Input((3, board_size, board_size))
    conv = Conv2D(256, (1, 1), padding='same', kernel_regularizer=l2(weight_decay))(input)
    batch_n = BatchNormalization()(conv)
    next_output = Activation('softplus')(batch_n)
    for i in range(num_blocks):
        next_output = add_skip_connection(next_output, input)
    flattened_input = Flatten()(next_output)
    policy_output = Dense(board_size ** 2, kernel_regularizer=l2(weight_decay))(flattened_input)
    policy_output = BatchNormalization()(policy_output)
    policy_output = Activation('sigmoid', name='policy_output')(policy_output)
    value_output = Dense(1, kernel_regularizer=l2(weight_decay))(flattened_input)
    value_output = BatchNormalization()(value_output)
    value_output = Activation('tanh', name='value_output')(value_output)
    model = Model(inputs=input, outputs=[value_output, policy_output])
    sgd = SGD(0.2, 0.9)
    model.compile(sgd, ['mean_squared_error', nn.softmax_cross_entropy_with_logits], metrics=['acc'])
    model.summary()
    return model


# adds a skip connection which adds the input to the output of a layer
def add_skip_connection(previous_output, input):
    output = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(previous_output)
    output = BatchNormalization()(output)
    output = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(output)
    output = BatchNormalization()(output)
    input = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(input)
    input = BatchNormalization()(input)
    output = Add()([output, input])
    return Activation('softplus')(output)
