import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate

def conv3d_block(input_tensor, num_filters):
    encoder = Conv3D(num_filters, (3, 3, 3), activation='relu', padding='same')(input_tensor)
    encoder = Conv3D(num_filters, (3, 3, 3), activation='relu', padding='same')(encoder)
    return encoder

def encoder_block(inputs, num_filters):
    encoder = conv3d_block(inputs, num_filters)
    encoder_pool = MaxPooling3D((2, 2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(inputs, concat_tensor, num_filters):
    decoder = UpSampling3D((2, 2, 2))(inputs)
    decoder = Concatenate(axis=-1)([decoder, concat_tensor])
    decoder = conv3d_block(decoder, num_filters)
    return decoder

def get_model(input_shape):
    inputs = Input(input_shape)
    encoder0_pool, encoder0 = encoder_block(inputs, 16)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 64)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 128)
    center = conv3d_block(encoder3_pool, 256)
    decoder3 = decoder_block(center, encoder3, 128)
    decoder2 = decoder_block(decoder3, encoder2, 64)
    decoder1 = decoder_block(decoder2, encoder1, 32)
    decoder0 = decoder_block(decoder1, encoder0, 16)
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(decoder0)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == "__main__":
    input_shape = (144, 192, 256, 1)
    model = get_model(input_shape)
    model.summary()
