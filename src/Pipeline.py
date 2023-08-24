import sys
sys.path.append('..')

import os
import cv2
import time
import datetime
import numpy as np
import src.CraftDetector as craft
import src.CraftTools as crafttools
import src.LSTMRecognizer as lstm
import src.LSTMTools as lstmtools
import src.YOLODetector as yolo
import src.YOLOBbox as yolobbox
import src.MediaHandling as mh
import src.XMLParser as xml

class Pipeline:
	"""A wrapper for a combination of Licence Plates Detector, Text Detector and Optical Character Recognizer.
	Args:
		vehicles: Vehicles detector (None currently)
		lp: Licence Plates Detector
		text: Text Detector
		ocr: Optical Character Recognizer
	"""
	def __init__(self, vehicles=None, lp=None, text=None, ocr=None):
		if lp is None:
			lp = yolo.Detector()
		if text is None:
			text = craft.Detector()
		if ocr is None:
			ocr = lstm.Recognizer()
		self.vehicles = vehicles
		self.lp = lp
		self.text = text
		self.ocr = ocr

	def run(self, images, syntax_analyzer=False, lp_kwargs=None, text_kwargs=None, ocr_kwargs=None):
		"""Run the pipeline on a list of multiples images.
		Args:
			images: The images to parse (list of filepaths, for now).
			*_kwargs: Keyboard arguments to pass to the keras.models.model.predict() method
		Returns:
			lp_boxes: List containing the bounding boxes of the licence plates in each image (YOLOv3 format)
			txt_pred: Predicted text inside those bounding boxes
		"""
		if lp_kwargs is None:
			lp_kwargs = {}
		if text_kwargs is None:
			text_kwargs = {}
		if ocr_kwargs is None:
			ocr_kwargs = {}

		txt_pred = []
		lp_boxes = self.lp.detect(images=images, **lp_kwargs)

		# sub_images: [[img,img,...],..., [img,img,...]]
		sub_images = [yolobbox.crop_boxes(image, boxes, ['licence_plate']) for image, boxes in zip(images,lp_boxes)]
		
		for roi in sub_images:
			if roi:
				images, scales = crafttools.scale_images(roi)
				txt_boxes = self.text.detect(images=images, **text_kwargs) #[lp,lp,...]
				txt_pred.append(self.ocr.recognize_from_boxes(images=images, box_groups=txt_boxes, syntax_analyzer=syntax_analyzer, **ocr_kwargs))
			if not roi:
				txt_pred.append([['']])
		return lp_boxes, txt_pred

	def image(self, images, syntax_analyzer=False, lp_kwargs=None, text_kwargs=None, ocr_kwargs=None, logs=False):
		"""Run the pipeline on a list of image paths.
		"""
		# Create object
		media = mh.MediaHandling(images)
		# Load image propierties
		frames = media.image_properties()
		# Log name
		log_name = '{}.xml'.format(datetime.datetime.now())  

		# Run Pipeline			
		lp_boxes, txt_pred =  self.run(frames, syntax_analyzer=syntax_analyzer, lp_kwargs=lp_kwargs, text_kwargs=text_kwargs, ocr_kwargs=ocr_kwargs)
	
		# Logs
		if logs:
			for path, dim, boxes, txt in zip(media.input, media.dim, lp_boxes, txt_pred):
				root_tree = xml.parse_root(input_path = path, dim = dim)
				image_tree = xml.parse_image(tree = root_tree, predictions = (boxes,txt), label='licence_plate', n_frame=None)
				xml.write_tree(tree = image_tree, output_path = log_name)
				print('Log saved in: {}'.format(log_name))

		return lp_boxes, txt_pred

	def video_frame(self, video, n_frame, syntax_analyzer=False, lp_kwargs=None, text_kwargs=None, ocr_kwargs=None, logs=False):
		"""Run the pipeline on a video frame.
		"""
		# Create object
		media = mh.MediaHandling(video)
		# Load video propierties
		capture = media.video_properties()
		# Frame set
		frame = media.frame_set(capture, n_frame)
		# Log name
		log_name = '{}.xml'.format(datetime.datetime.now())  

		# Run Pipeline			
		lp_boxes, txt_pred =  self.run([frame], syntax_analyzer=syntax_analyzer, lp_kwargs=lp_kwargs, text_kwargs=text_kwargs, ocr_kwargs=ocr_kwargs)
	
		# Logs
		if logs:
			for boxes, txt in zip(lp_boxes, txt_pred):
				root_tree = xml.parse_root(input_path = media.input, dim = media.dim)
				image_tree = xml.parse_image(tree=root_tree, predictions = (boxes,txt), label='licence_plate', n_frame=n_frame)
				xml.write_tree(tree = image_tree, output_path = log_name)
				print('Log saved in: {}'.format(log_name))

		# Capture release & return
		capture.release()
		return lp_boxes, txt_pred
	
	def full_video(self, video, file_output=None, syntax_analyzer=False, lp_kwargs=None, text_kwargs=None, ocr_kwargs=None, logs=True):
		"""Run  and print the pipeline on a full video.
		"""
		# Create object
		media = mh.MediaHandling(video)
		# Load video propierties
		capture = media.video_properties()
		# Log name & root
		if logs:
			log_name = '{}.xml'.format(datetime.datetime.now())  
			tree = xml.parse_root(input_path = media.input, dim = media.dim)

		# Image Array to form a MP4 video & current frame
		n_frame = 0
		img_array = []

		# Inference Loop
		while(n_frame<media.frames):

			# Capture read
			ret, frame, n_frame = media.read_frame(capture, current_frame=True)
			# Debugging
			print('Working on frame: {}'.format(n_frame))
			print('ret: {}'.format(ret))

			# Run Pipeline			
			lp_boxes, txt_pred =  self.run([frame], syntax_analyzer=syntax_analyzer, lp_kwargs=lp_kwargs, text_kwargs=text_kwargs, ocr_kwargs=ocr_kwargs)

			# New video
			if file_output:
				# ROI & Label
				for boxes in lp_boxes:
					frame_output = yolobbox.draw_boxes(frame, boxes, ['licence_plate'])

				#Append segmented frame to Image Array
				img_array.append(frame_output)
				
			# Logs
			if logs:
				for boxes, txt in zip(lp_boxes, txt_pred):
					tree = xml.parse_image(tree=tree, predictions = (boxes,txt), label='licence_plate', n_frame=n_frame)

		# Video save
		if file_output:
			media.video_save(file_output, img_array)

		# Logs save
		if logs:
			xml.write_tree(tree = tree, output_path = log_name)
			print('Log saved in: {}'.format(log_name))

		# Capture release & return
		capture.release()
		return None

if __name__ == '__main__':
	pass