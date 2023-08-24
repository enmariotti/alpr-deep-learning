import sys
sys.path.append('..')

import numpy as np
import src.CraftTools as tools
import src.VGGModel as vggmodel

import keras

class Detector:
	"""A text detector using the CRAFT architecture.
	"""
	def __init__(self, weights_path='models/craft_mlt_25k.h5', backbone_name='vgg'):
		self.model = vggmodel.build_keras_model(weights_path=weights_path, backbone_name=backbone_name)

	def detect(self, images,
			   detection_threshold=0.8,
			   text_threshold=0.25,
			   link_threshold=0.7,
			   size_threshold=10,
			   rescale_boxes = False,
			   **kwargs):
		"""Recognize the text in a set of images.
		Args:
			images: Can be a list of numpy arrays of shape HxWx3.
			link_threshold: This is the same as `text_threshold`, but is applied to the
				link map instead of the text map.
			detection_threshold: We want to avoid including boxes that may have
				represented large regions of low confidence text predictions. To do this,
				we do a final check for each word box to make sure the maximum confidence
				value exceeds some detection threshold. This is the threshold used for
				this check.
			text_threshold: When the text map is processed, it is converted from confidence
				(float from zero to one) values to classification (0 for not text, 1 for
				text) using binary thresholding. The threshold value determines the
				breakpoint at which a value is converted to a 1 or a 0. For example, if
				the threshold is 0.4 and a value for particular point on the text map is
				0.5, that value gets converted to a 1. The higher this value is, the less
				likely it is that characters will be merged together into a single word.
				The lower this value is, the more likely it is that non-text will be detected.
				Therein lies the balance.
			size_threshold: The minimum area for a word.
		"""
		images = [tools.compute_input(image) for image in images]
		boxes = tools.getBoxes(self.model.predict(np.array(images), **kwargs),
						 detection_threshold=detection_threshold,
						 text_threshold=text_threshold,
						 link_threshold=link_threshold,
						 size_threshold=size_threshold)
		return boxes