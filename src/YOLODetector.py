import sys
sys.path.append('..')

import os
import keras

import src.YOLOTools as tools

class Detector:
	"""A text Licence Plate Detector using the YOLOv3 architecture.
	"""
	def __init__(self, weights_path='models/lp_yolov3.h5', backbone_name='yolov3'):
		if backbone_name == 'yolov3':
			model = keras.models.load_model(weights_path, compile=False) # No custom layers on this model
		else:
			raise NotImplementedError
		self.model = model

	def detect(self, images,
			   net_h=416,
			   net_w=416,
			   obj_thresh=0.75,
			   nms_thresh=0.45,
			   anchors=[15,6, 18,8, 22,9, 27,11, 32,13, 41,17, 54,21, 66,27, 82,33],
			   **kwargs):
		"""Recognize the text in a set of images.
		Args:
			images: A list of numpy arrays of shape HxWx3.
			net_h: Input heigth.
			net_w: Input width.
			obj_thresh: Threshold on P(object).
			nms_thresh: Non-maximal Supression Threshold.
			anchors: Classes anchors. Default are European LP anchors.
		"""
		boxes = tools.get_yolo_boxes(model=self.model, images=images,
				 net_h=net_h,
				 net_w=net_w,
				 anchors=anchors,
				 obj_thresh=obj_thresh,
				 nms_thresh=nms_thresh, **kwargs)

		return boxes