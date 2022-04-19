import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout,LSTM, Multiply, Flatten, Reshape, Conv2D, TimeDistributed, LeakyReLU, Input, GlobalAveragePooling2D, concatenate,Dense, MaxPooling2D,AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import he_uniform
from os.path import join


def FlowNetModule(input):
    # net = AveragePooling2D(strides=2,padding="same")(input)
    net = Conv2D(64, 7, strides=(2, 2), padding='same')(input)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(128, 5, strides=(2, 2), padding='same')(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(256, 5,  strides=(2, 2), padding='same')(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(256, 3,  strides=(1, 1), padding='same')(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(512, 3,  strides=(2, 2), padding='same')(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(512, 3, strides=(1, 1), padding='same')(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(512, 3, strides=(2, 2), padding='same')(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(512, 3, strides=(1, 1), padding='same')(net)
    net = LeakyReLU(alpha=0.1)(net)
    net = Conv2D(1024, 3, strides=(2, 2), padding='same')(net)
    net = LeakyReLU(alpha=0.1)(net)
    # net = Conv2D(1024,3,strides=(2, 2), padding='same')(net)
    # net = LeakyReLU(alpha=0.1)(net)
    # net = MaxPooling2D(strides=2,padding="same")(net)
    return net

def build_model_cross_att(cfg, imu_length=30, input_shape=(64, 256, 1), mask_att='sigmoid', istraining=True, write_mask=False):
        image_1 = Input(name='image_1',shape=input_shape)
        image_2 = Input(name='image_2',shape=input_shape)
        image_merged = concatenate([image_1,image_2])
        net = FlowNetModule(image_merged) # 1,4,1024 -> 2,8,1024
        
        visual_mask = (GlobalAveragePooling2D())(net) # reshape to (?, 1, 1024), 1 stands for timeDistr.
        visual_mask = (Dense(int(1024/256), activation='relu', use_bias=False, name='visual_mask_relu'))(visual_mask)
        visual_mask = (Dense(1024, activation='sigmoid', use_bias=False, name='visual_mask_sigmoid'))(visual_mask)
        visual_mask = Reshape((1, 1, 1024))(visual_mask)

        # activate mask by element-wise multiplication
        visual_att_fea = Multiply()([net, visual_mask]) 
        visual_att_fea = (Flatten())(visual_att_fea)

        # IMU data
        imu_data = Input(shape=(imu_length, 6), name='imu_data')
        imu_lstm_1 = LSTM(128, return_sequences=True, name='imu_lstm_1')(imu_data)  # 128, 256

        # channel-wise IMU attention
        reshape_imu = Reshape((1, imu_length * 128))(imu_lstm_1)  # 2560, 5120, 10240
        imu_mask = Dense(128, activation='relu', use_bias=False, name='imu_mask_relu')(reshape_imu)
        imu_mask = Dense(imu_length * 128, activation='sigmoid', use_bias=False, name='imu_mask_sigmoid')(imu_mask)
        imu_att_fea = Multiply()([reshape_imu, imu_mask])

        # cross-modal attention
        imu4visual_mask = Dense(128, activation='relu', use_bias=False, name='imu4visual_mask_relu')(imu_att_fea)
        imu4visual_mask = Dense(4096, activation=mask_att, use_bias=False, name='imu4visual_mask_sigmoid')(imu4visual_mask)
        cross_visual_fea = Multiply()([visual_att_fea, imu4visual_mask])

        visual4imu_mask = Dense(128, activation='relu', use_bias=False, name='visual4imu_mask_relu')(visual_att_fea)
        visual4imu_mask = Dense(imu_length * 128, activation=mask_att, use_bias=False, name='visual4imu_mask_sigmoid')(visual4imu_mask)
        cross_imu_fea = Multiply()([imu_att_fea, visual4imu_mask])

        # Standard merge feature
        merge_features = concatenate([cross_visual_fea, cross_imu_fea], axis=-1, name='merge_features')

        # Selective features
        forward_lstm_1 = LSTM(512, dropout=0.25, return_sequences=True, name='forward_lstm_1')(
            merge_features)  # dropout_W=0.2, dropout_U=0.2
        forward_lstm_2 = LSTM(512, return_sequences=True, name='forward_lstm_2')(forward_lstm_1)

        fc_position_1 = (Dense(128, activation='relu'))(forward_lstm_2)  # tanh
        dropout_pos_1 = (Dropout(0.25))(fc_position_1)
        fc_position_2 = (Dense(64, activation='relu'))(dropout_pos_1)  # tanh
        fc_trans = (Dense(3,name = 'fc_trans'))(fc_position_2)

        fc_orientation_1 = (Dense(128, activation='relu', name='fc_orientation_1'))(forward_lstm_2)  # tanh
        dropout_orientation_1 = (Dropout(0.25, name='dropout_wpqr_1'))(fc_orientation_1)
        fc_orientation_2 = (Dense(64, activation='relu', name='fc_orientation_2'))(
            dropout_orientation_1)  # tanh
        fc_rot = (Dense(3, name='fc_rot'))(fc_orientation_2)

        model = Model(inputs=[image_1, image_2,imu_data], outputs=[fc_trans, fc_rot])
        #debugging:
        # if istraining == False:   
        #      model = Model(inputs=[image_1, image_2, imu_data], outputs=[fc_trans, fc_rot])
        

        return model
