"""This is where you design your model
"""

from tensorflow.keras import Model, layers

class Forcaster(Model):

    def __init__(self, input_shape, output_shape, units=32):
        super(Forcaster, self).__init__()
        self.input_layer = layers.LSTM(units, return_sequences= True, input_shape = input_shape)
        self.drop_out_1 = layers.Dropout(0.1)
        self.hidden_layer_1 = layers.LSTM(units)
        self.drop_out_2 = layers.Dropout(0.1)
        self.output_layer = layers.Dense(output_shape)

    def call(self, x, training=False):
        x = self.input_layer(x)
        if training:
            x = self.drop_out_1(x, training=training)
        x = self.hidden_layer_1(x)
        if training:
            x = self.drop_out_2(x, training=training)
        return self.output_layer(x)


