# LSTMNet.py
# IB - MS Engineering
# Developer: Enrique Nicanor Mariotti (mariottien@gmail.com)
#
# LSTM Model predictor - CTC Decoder
# Load Conda environment 'cnn_environment.yml' to properly execute. 
import sys
sys.path.append('..')

import os
import cv2
import argparse
import numpy as np

import keras
import src.LSTMTools as tools
import src.LSTMModels as lstmmodels

import matplotlib.pyplot as plt

class Recognizer():
	"""An OCR using the bidirectional CNN backbone, Spatial Transformers and LSTM layers.
	"""
	def __init__(self, weights_path='models/biLSTM_stn.hdf5', backbone_name='biLSTMSTN'):
		self.model, self.alphabet = lstmmodels.build_keras_model(weights_path=weights_path, backbone_name=backbone_name)

	def recognize_from_boxes(self, images, box_groups, syntax_analyzer=False, **kwargs):
		"""Recognize text from images using lists of bounding boxes from CRAFT.
		Args:
			images: A list of input images, supplied as numpy arrays with shape
				(H, W, 3).
			boxes: A list of groups of boxes, one for each image
		"""
		assert len(box_groups) == len(images), \
			'You must provide the same number of box groups as images.'
		if syntax_analyzer:
			box_groups = self.boxes_hierarchy(images=images, txt_boxes=box_groups)
		
		crops = []
		for image, boxes in zip(images, box_groups): #[lp,lp,...]
			# words: [word,word,...] 
			words = [tools.warpBox(image=image, box=box, target_height=None, target_width=None) for box in boxes]
			# crops: [[word,word,...],..., [word,word,...]] 
			crops.append(words)
		if not crops: #[lp_img, lp_img]
			predictions = [[''] for image in images]
		if crops:
			predictions = [self.recognize(images=words, **kwargs) if words else [''] for words in crops] #[word,word,...]
		return predictions

	def boxes_hierarchy(self, images, txt_boxes):
		areas = [int(image.shape[0]*image.shape[1]) for image in images]
		boxes_list = []
		# words_list = []    
		for boxes, image_area in zip(txt_boxes, areas):
			contour_area = np.array([cv2.contourArea(contour)/image_area for contour in boxes]) 
			idx_sort = np.flip(np.argsort(contour_area, axis=None), axis=None)
			#sorting
			contour_area = contour_area[idx_sort]
			boxes = boxes[idx_sort]
			#threshold
			idx_threshold = contour_area>=0.05
			boxes = boxes[idx_threshold]
			contour_area = contour_area[idx_threshold]
			# words_count = boxes.shape[0]
			if boxes.size == 0:
				boxes_sort = boxes
			if boxes.size != 0:
				boxes_sort = self.boxes_readable(boxes=boxes)
			boxes_list.append(boxes_sort)
			# words_list.append(words_count)
		return boxes_list

	def boxes_readable(self, boxes):
		# Points sorting. To get min norm point first
		norm_points = np.linalg.norm(boxes, axis=2)
		idx_norm = np.argsort(norm_points)
		boxes_norm = boxes[:,idx_norm[0,:],:]
		# Get min (x, y)
		xy_min = boxes_norm[:,0,:]
		# Box Sorting
		norm_min = np.linalg.norm(xy_min, axis=1)
		idx_min = np.argsort(norm_min)
		boxes_sort = boxes[idx_min,:,:]
		return boxes_sort

	def recognize(self, images, **kwargs):
		images = [tools.preProcessing(image) for image in images]
		predictions = self.model.predict(np.array(images), **kwargs) #[word,word,...]
		shape = predictions[:, 2:, :].shape
		ctc_decode = keras.backend.ctc_decode(predictions[:, 2:, :], input_length=np.ones(shape[0])*shape[1], greedy=True)
		ctc_out = keras.backend.get_value(ctc_decode[0][0])
		# String Decoder
		decoded_predictions = []
		for ctc in ctc_out:
			result_str = ''.join([self.alphabet[c] for c in ctc])
			result_str = result_str.replace('-', '')
			decoded_predictions.append(result_str)
		return decoded_predictions