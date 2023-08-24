import cv2
import string
import numpy as np
from scipy import spatial
from shapely import geometry

def syntax_analyzer(txt_boxes, images):
    return txt_boxes

def char_list_generator(lowercase=False):
	if lowercase == False:
		char_list = string.digits+string.ascii_letters+'-'
	if lowercase == True:
		char_list = string.digits+string.ascii_lowercase+'-'	
	char_list_len = len(char_list)
	return char_list, char_list_len 

def preProcessing(image, padded_only=False):

	# Expected ref size, ratio and background color
	ref_size = (31, 200) #(height, width)
	ref_ratio = float(ref_size[0]/ref_size[1])
	pad_color = [0, 0, 0]

	# Read image & RGB to GRAYS
	try:
		if image.shape[-1] == 3:
			im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		original_size = im.shape[:2] # (height, width)
		original_ratio = float(original_size[0]/original_size[1])
	except:
		raise ValueError('ValueError: CV2 could not convert to Grayscale.')

	if padded_only==True: 
		# Margins if sizes matches
		top, bottom = 0, 0
		left, right = 0, 0

		if original_size[0] > ref_size[0] or original_size[1] > ref_size[1]:
			raise ValueError('ValueError: One or more dimensions exceed (h, w) = (31, 200). The "padded_only" format cannot be applied.')
		
		if original_size[0] < ref_size[0]:
			delta_h = ref_size[0] - original_size[0]
			top, bottom = 0, delta_h
		
		if original_size[1] < ref_size[1]:
			delta_w = ref_size[1] - original_size[1]
			left, right = 0, delta_w
		
		padded_image = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
		padded_image = np.transpose(padded_image)
		padded_image = np.flip(padded_image,1)	
		padded_image = np.expand_dims(padded_image, axis = 2)
		padded_image = padded_image/255.
		return padded_image
		
	if padded_only==False:

		if original_ratio == ref_ratio: 
			top, bottom = 0, 0
			left, right = 0, 0

		if original_ratio < ref_ratio: 
			delta_h = int(original_size[1]*ref_ratio) - original_size[0]
			top, bottom = 0, delta_h
			left, right = 0, 0

		if original_ratio > ref_ratio: 
			delta_w = int(original_size[0]*(1./ref_ratio)) - original_size[1]
			top, bottom = 0, 0
			left, right = 0, delta_w

		padded_image = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
		output_image = cv2.resize(padded_image, (ref_size[1], ref_size[0]), interpolation=cv2.INTER_NEAREST) # (width, height)
		output_image = np.transpose(output_image)
		output_image = np.flip(output_image,1)
		output_image = np.expand_dims(output_image, axis = 2)
		output_image = output_image/255.
		return output_image

def warpBox(image,
            box,
            target_height=None,
            target_width=None,
            margin=0,
            cval=None,
            return_transform=False,
            skip_rotate=False):
    """Warp a boxed region in an image given by a set of four points into
    a rectangle with a specified width and height. Useful for taking crops
    of distorted or rotated text.
    Args:
        image: The image from which to take the box
        box: A list of four points starting in the top left
            corner and moving clockwise.
        target_height: The height of the output rectangle
        target_width: The width of the output rectangle
        return_transform: Whether to return the transformation
            matrix with the image.
    """
    if cval is None:
        cval = (0, 0, 0) if len(image.shape) == 3 else 0
    if not skip_rotate:
        box, _ = get_rotated_box(box)
    w, h = get_rotated_width_height(box)
    assert (
        (target_width is None and target_height is None)
        or (target_width is not None and target_height is not None)), \
            'Either both or neither of target width and height must be provided.'
    if target_width is None and target_height is None:
        target_width = w
        target_height = h
    scale = min(target_width / w, target_height / h)
    M = cv2.getPerspectiveTransform(src=box,
                                    dst=np.array([[margin, margin], [scale * w - margin, margin],
                                                  [scale * w - margin, scale * h - margin],
                                                  [margin, scale * h - margin]]).astype('float32'))
    crop = cv2.warpPerspective(image, M, dsize=(int(scale * w), int(scale * h)))
    target_shape = (target_height, target_width, 3) if len(image.shape) == 3 else (target_height,
                                                                                   target_width)
    full = (np.zeros(target_shape) + cval).astype('uint8')
    full[:crop.shape[0], :crop.shape[1]] = crop
    if return_transform:
        return full, M
    return full

def get_rotated_box(points):
    """Obtain the parameters of a rotated box.
    Returns:
        The vertices of the rotated box in top-left,
        top-right, bottom-right, bottom-left order along
        with the angle of rotation about the bottom left corner.
    """
    try:
        mp = geometry.MultiPoint(points=points)
        pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[:-1]  # noqa: E501
    except AttributeError:
        # There weren't enough points for the minimum rotated rectangle function
        pts = points
    # The code below is taken from
    # https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    pts = np.array([tl, tr, br, bl], dtype="float32")

    rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
    return pts, rotation

def get_rotated_width_height(box):
    """
    Returns the width and height of a rotated rectangle
    Args:
        box: A list of four points starting in the top left
        corner and moving clockwise.
    """
    w = (spatial.distance.cdist(box[0][np.newaxis], box[1][np.newaxis], "euclidean") +
         spatial.distance.cdist(box[2][np.newaxis], box[3][np.newaxis], "euclidean")) / 2
    h = (spatial.distance.cdist(box[0][np.newaxis], box[3][np.newaxis], "euclidean") +
         spatial.distance.cdist(box[1][np.newaxis], box[2][np.newaxis], "euclidean")) / 2
    return int(w[0][0]), int(h[0][0])