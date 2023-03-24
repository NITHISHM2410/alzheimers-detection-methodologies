class SQD(tf.keras.layers.Layer):
    def __init__(self):
        super(SQD, self).__init__()
    def call(self, inputs):
        x, y = inputs
        diff = tf.subtract(x, y)
        return tf.square(diff)
        

def wave_downsample():
    input = tf.keras.Input(batch_size = batch_size,shape = (1024,19))
    output = tf.keras.layers.LSTM(10,return_sequences = True)(input)
    output = tf.keras.layers.Dense(2)(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Activation('relu')(output)
    output = tf.keras.layers.Dropout(0.3)(output)
    output = tf.keras.layers.Dense(20)(output)
    output = tf.keras.layers.Dropout(0.1)(output)
    model = tf.keras.Model(inputs = [input],outputs = [output])
    return model


def get_model():
    wave1 = tf.keras.Input(batch_size = batch_size,shape = (1024,19),name = 'wave1')
    wave2 = tf.keras.Input(batch_size = batch_size,shape = (1024,19),name = 'wave2')
    state = tf.keras.Input(batch_size = batch_size,shape = (2,),name = 'state')
    
    output1 = tf.keras.layers.BatchNormalization()(wave1)
    output2 = tf.keras.layers.BatchNormalization()(wave2)

    down_sampler = wave_downsample()
    output1 = down_sampler(output1)
    output2 = down_sampler(output2)

    output1 = tf.keras.layers.Activation('tanh')(output1)
    output1 = tf.keras.layers.Dropout(0.3)(output1)

    output2 = tf.keras.layers.Activation('tanh')(output2)
    output2 = tf.keras.layers.Dropout(0.3)(output2)

    output = SQD()([output1,output2])
    output = tf.keras.layers.concatenate([output,state])
    output = tf.keras.layers.Dense(10)(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(1,activation = 'sigmoid',name = 'difference' )(output)
    
    model = tf.keras.Model(inputs = [wave1,wave2,state],outputs = [output],name = 'FewShot')
    return model
