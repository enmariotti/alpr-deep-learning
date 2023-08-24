import cv2
import numpy as np

def drawBoxes(image, boxes, color=(255, 0, 0), thickness=5, boxes_format='boxes'):
	"""Draw boxes onto an image.
	Args:
		image: The image on which to draw the boxes.
		boxes: The boxes to draw.
		color: The color for each box.
		thickness: The thickness for each box.
		boxes_format: The format used for providing the boxes. Options are
			"boxes" which indicates an array with shape(N, 4, 2) where N is the
			number of boxes and each box is a list of four points) as provided
			by `keras_ocr.detection.Detector.detect`, "lines" (a list of
			lines where each line itself is a list of (box, character) tuples) as
			provided by `keras_ocr.data_generation.get_image_generator`,
			or "predictions" where boxes is by itself a list of (word, box) tuples
			as provided by `keras_ocr.pipeline.Pipeline.recognize` or
			`keras_ocr.recognition.Recognizer.recognize_from_boxes`.
	"""
	if len(boxes) == 0:
		return image
	canvas = image.copy()
	if boxes_format == 'lines':
		revised_boxes = []
		for line in boxes:
			for box, _ in line:
				revised_boxes.append(box)
		boxes = revised_boxes
	if boxes_format == 'predictions':
		revised_boxes = []
		for _, box in boxes:
			revised_boxes.append(box)
		boxes = revised_boxes
	for box in boxes:
		cv2.polylines(img=canvas,
					  pts=box[np.newaxis].astype('int32'),
					  color=color,
					  thickness=thickness,
					  isClosed=True)
	return canvas

def adjust_boxes(boxes, boxes_format='boxes', scale=1):
	"""Adjust boxes using a given scale and offset.
	Args:
		boxes: The boxes to adjust
		boxes_format: The format for the boxes. See the `drawBoxes` function
			for an explanation on the options.
		scale: The scale to apply
	"""
	if scale == 1:
		return boxes
	if boxes_format == 'boxes':
		return np.array(boxes) * scale
	if boxes_format == 'lines':
		return [[(np.array(box) * scale, character) for box, character in line] for line in boxes]
	if boxes_format == 'predictions':
		return [(word, np.array(box) * scale) for word, box in boxes]
	raise NotImplementedError(f'Unsupported boxes format: {boxes_format}')

def resize_image(image, max_scale, max_size):
	"""Obtain the optimal resized image subject to a maximum scale
	and maximum size.
	Args:
		image: The input image
		max_scale: The maximum scale to apply
		max_size: The maximum size to return
	"""
	if max(image.shape) * max_scale > max_size:
		# We are constrained by the maximum size
		scale = max_size / max(image.shape)
	else:
		# We are contrained by scale
		scale = max_scale
	return cv2.resize(image, dsize=(int(image.shape[1] * scale), int(image.shape[0] * scale))), scale

def scale_images(images, max_scale=2, max_size=2048):
	images = [resize_image(image, max_scale=max_scale, max_size=max_size) for image in images if image.size != 0]
	max_height, max_width = np.array([image.shape[:2] for image, scale in images]).max(axis=0)
	scales = [scale for _, scale in images]
	images = np.array([pad(image, width=max_width, height=max_height) for image, _ in images])
	return images, scales

def rescale_box(box_groups, scales):
	box_groups = [adjust_boxes(boxes=boxes, boxes_format='boxes', scale=1/scale) 
		if scale != 1 else boxes for boxes, scale in zip(box_groups, scales)]
	return box_groups	

def compute_input(image):
	# should be RGB order
	image = image.astype('float32')
	mean = np.array([0.485, 0.456, 0.406])
	variance = np.array([0.229, 0.224, 0.225])

	image -= mean * 255
	image /= variance * 255
	return image

def pad(image, width: int, height: int, cval: int = 255):
	"""Pad an image to a desired size. Raises an exception if image
	is larger than desired size.
	Args:
		image: The input image
		width: The output width
		height: The output height
		cval: The value to use for filling the image.
	"""
	if len(image.shape) == 3:
		output_shape = (height, width, image.shape[-1])
	else:
		output_shape = (height, width)
	assert height >= output_shape[0], 'Input height must be less than output height.'
	assert width >= output_shape[1], 'Input width must be less than output width.'
	padded = np.zeros(output_shape, dtype=image.dtype) + cval
	padded[:image.shape[0], :image.shape[1]] = image
	return padded

def getBoxes(y_pred,
			 detection_threshold=0.7,
			 text_threshold=0.4,
			 link_threshold=0.4,
			 size_threshold=10):
	box_groups = []
	for y_pred_cur in y_pred:
		# Prepare data
		textmap = y_pred_cur[..., 0].copy()
		linkmap = y_pred_cur[..., 1].copy()
		img_h, img_w = textmap.shape

		_, text_score = cv2.threshold(textmap,
									  thresh=text_threshold,
									  maxval=1,
									  type=cv2.THRESH_BINARY)
		_, link_score = cv2.threshold(linkmap,
									  thresh=link_threshold,
									  maxval=1,
									  type=cv2.THRESH_BINARY)
		n_components, labels, stats, _ = cv2.connectedComponentsWithStats(np.clip(
			text_score + link_score, 0, 1).astype('uint8'),
																		  connectivity=4)
		boxes = []
		for component_id in range(1, n_components):
			# Filter by size
			size = stats[component_id, cv2.CC_STAT_AREA]

			if size < size_threshold:
				continue

			# If the maximum value within this connected component is less than
			# text threshold, we skip it.
			if np.max(textmap[labels == component_id]) < detection_threshold:
				continue

			# Make segmentation map. It is 255 where we find text, 0 otherwise.
			segmap = np.zeros_like(textmap)
			segmap[labels == component_id] = 255
			segmap[np.logical_and(link_score, text_score)] = 0
			x, y, w, h = [
				stats[component_id, key] for key in
				[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
			]

			# Expand the elements of the segmentation map
			niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
			sx, sy = max(x - niter, 0), max(y - niter, 0)
			ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
			segmap[sy:ey, sx:ex] = cv2.dilate(
				segmap[sy:ey, sx:ex],
				cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)))

			# Make rotated box from contour
			contours = cv2.findContours(segmap.astype('uint8'),
										mode=cv2.RETR_TREE,
										method=cv2.CHAIN_APPROX_SIMPLE)[-2]
			contour = contours[0]
			box = cv2.boxPoints(cv2.minAreaRect(contour))

			# Check to see if we have a diamond
			w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
			box_ratio = max(w, h) / (min(w, h) + 1e-5)
			if abs(1 - box_ratio) <= 0.1:
				l, r = contour[:, 0, 0].min(), contour[:, 0, 0].max()
				t, b = contour[:, 0, 1].min(), contour[:, 0, 1].max()
				box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
			else:
				# Make clock-wise order
				box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
			boxes.append(2 * box)
		box_groups.append(np.array(boxes))
	return box_groups
