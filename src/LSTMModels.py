import sys
sys.path.append('..')

import os
import numpy as np
import keras
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # {'0', '1', '2', '3'}
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = 'off'

import src.STNLayers as stn
import src.LSTMTools as tools

def build_keras_model(weights_path, backbone_name='biLSTMSTN'):
	if backbone_name == 'biLSTMSTN':
		char_list, char_list_len = tools.char_list_generator(lowercase=True)
		model = biLSTMModel_STN(char_list_len)
	else:
		raise NotImplementedError
	if weights_path is not None and weights_path.endswith('.hdf5'):
		model.load_weights(weights_path)
	else:
		raise NotImplementedError(f'Cannot load weights from {weights_path}')
	return model, char_list

def LOC_Net(input_shape):
	b = np.zeros((2, 3), dtype='float32')
	b[0, 0] = 1
	b[1, 1] = 1
	w = np.zeros((64, 6), dtype='float32')
	weights = [w, b.flatten()]

	loc_input = keras.layers.Input(input_shape)
	loc_conv_1 = keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(loc_input)
	loc_conv_2 = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(loc_conv_1)
	loc_fla = keras.layers.Flatten()(loc_conv_2)
	loc_fc_1 = keras.layers.Dense(64, activation='relu')(loc_fla)
	loc_fc_2 = keras.layers.Dense(6, weights=weights)(loc_fc_1)

	output = keras.models.Model(inputs=loc_input, outputs=loc_fc_2)
	return output

def biLSTMModel_STN(char_list_len):
	CHAR_LIST_LENGTH_EPS = char_list_len

	# keras.layers.Input layer (200,31,1) Grayscale
	input_layer = keras.layers.Input(shape=(200,31,1))

	# CNN keras.models.Model
	# conv_1 = keras.layers.Conv2D(64, (3,3), activation = 'relu', padding='same')(input_layer)
	# pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)	 

	conv_1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1')(input_layer)
	conv_2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2')(conv_1)
	conv_3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3')(conv_2)
	batch_norm_3 = keras.layers.BatchNormalization(name='bn_3')(conv_3)
	pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2), name='maxpool_3')(batch_norm_3)

	conv_4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_4')(pool_3)
	conv_5 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_5')(conv_4)
	batch_norm_5 = keras.layers.BatchNormalization(name='bn_5')(conv_5)
	pool_5 = keras.layers.MaxPool2D(pool_size=(2, 2), name='maxpool_5')(batch_norm_5)

	conv_6 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_6')(pool_5)
	conv_7 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_7')(conv_6)
	batch_norm_7 = keras.layers.BatchNormalization(name='bn_7')(conv_7)

	conv_output_shape = batch_norm_7.get_shape()
	LOC_input_shape = (conv_output_shape[1].value, conv_output_shape[2].value, conv_output_shape[3].value)
	STN = stn.SpatialTransformer(localization_net=LOC_Net(LOC_input_shape), output_size=(LOC_input_shape[0], LOC_input_shape[1]))(batch_norm_7)

	# keras.layers.Reshape & keras.layers.Dense Layer 
	reshape_layer = keras.layers.Reshape(target_shape=(int(conv_output_shape[1]), int(conv_output_shape[2] * conv_output_shape[3])), name='reshape_layer')(STN)
	dense_layer_9 = keras.layers.Dense(128, activation='relu', name='fc_9')(reshape_layer)

	# keras.layers.Bidirectional keras.layers.LSTM layers with units=128
	lstm_10 = keras.layers.LSTM(128, kernel_initializer="he_normal", return_sequences=True, name='lstm_10')(dense_layer_9)
	lstm_10_back = keras.layers.LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True, name='lstm_10_back')(dense_layer_9)
	lstm_10_add = keras.layers.add([lstm_10, lstm_10_back])

	lstm_11 = keras.layers.LSTM(128, kernel_initializer="he_normal", return_sequences=True, name='lstm_11')(lstm_10_add)
	lstm_11_back = keras.layers.LSTM(128, kernel_initializer="he_normal", go_backwards=True, return_sequences=True, name='lstm_11_back')(lstm_10_add)
	lstm_11_concat = keras.layers.concatenate([lstm_11, lstm_11_back])
	dropout_11 = keras.layers.Dropout(0.25, name='dropout_11')(lstm_11_concat)

	# Return keras.models.Model
	output_layer = keras.layers.Dense(CHAR_LIST_LENGTH_EPS, kernel_initializer='he_normal', activation='softmax', name='fc_12')(dropout_11)

	# Test keras.models.Model
	test_model = keras.models.Model(inputs=input_layer, outputs=output_layer)

	return test_model