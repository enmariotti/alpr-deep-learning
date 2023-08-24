# MediaHandling.py
# IB - MS Engineering
# Developer: Enrique Nicanor Mariotti (mariottien@gmail.com)
#
# Media Handling library using Open CV for segmentation module.
# Load Conda environment 'environment.yml' to properly execute. 

import cv2
import numpy as np
import matplotlib.pyplot as plt

class MediaHandling():

	def __init__(self, input_media):
		self.input = input_media

	def image_properties(self): 
		frames = [cv2.imread(image,cv2.IMREAD_COLOR) for image in self.input] #always as BRG image
		self.height = [int(frame.shape[0]) for frame in frames]
		self.width  = [int(frame.shape[1]) for frame in frames]
		self.dim = [(w, h) for w, h in zip(self.width, self.height)]
		return frames

	def video_properties(self): 
		capture = cv2.VideoCapture(self.input)
		self.frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
		self.width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps	= int(capture.get(cv2.CAP_PROP_FPS))
		self.dim = (self.width, self.height)
		return capture

	def webcam_properties(self): 
		capture = cv2.VideoCapture(self.input)
		self.width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps	= int(capture.get(cv2.CAP_PROP_FPS))
		self.dim = (self.width, self.height)
		return capture

	def frame_set(self, capture, n_frame):
		capture.set(1,n_frame)
		ret, frame, current_frame = self.read_frame(capture)
		return frame

	def read_frame(self, capture, current_frame=False): 
		ret, frame = capture.read()
		if current_frame:
			current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
		if not current_frame:
			current_frame = None
		return ret, frame, current_frame

	def image_save(self, file_name, image): 
		cv2.imwrite(file_name, image)

	def video_save(self, file_name, img_array):
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		output_video = cv2.VideoWriter('{}.avi'.format(file_name),fourcc, self.fps, self.dim)
		for frame in img_array:
			output_video.write(frame)
		output_video.release()

if __name__ == "__main__":
	pass