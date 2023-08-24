import os
import sys
import xml.etree.ElementTree as ET

def parse_root(input_path, dim):
	prediction = ET.Element('prediction')

	# File path
	folder = ET.SubElement(prediction, 'folder')
	folder.text = os.path.basename(os.getcwd()) 
	filename = ET.SubElement(prediction, 'filename')
	filename.text = os.path.split(input_path)[1]
	path = ET.SubElement(prediction, 'path')
	path.text = os.path.abspath(input_path)

	# Dataset
	source = ET.SubElement(prediction, 'source')
	database = ET.SubElement(source, 'database')
	database.text = 'Unknown'

	# Image propierties
	size = ET.SubElement(prediction, 'size')
	width = ET.SubElement(size, 'width')
	width.text = str(dim[0])
	height = ET.SubElement(size, 'height')
	height.text = str(dim[1])
	depth = ET.SubElement(size, 'depth')
	depth.text = '3'

	# Segmented
	segmented = ET.SubElement(prediction, 'segmented')
	segmented.text = '0'

	root_indent(prediction)
	tree = ET.ElementTree(prediction)
	return tree

def parse_image(tree, predictions, label='licence_plate', n_frame=None):
	# predictions=(bbox, txt)
	root = tree.getroot()

	# Frame
	frame = ET.SubElement(root, 'frame')
	frame_number = ET.SubElement(frame, 'frame_number')	
	frame_number.text = str(n_frame)

	if predictions:
		boxes = predictions[0]
		id_txt = predictions[1]
		for bbox, txt in zip(boxes, id_txt):
			lp = ET.SubElement(frame, 'object')

			# Miscellaneous
			name = ET.SubElement(lp, 'name')
			name.text = label
			pose = ET.SubElement(lp, 'pose')
			pose.text = 'Unspecified'
			truncated = ET.SubElement(lp, 'truncated')
			truncated.text = '0'
			difficult = ET.SubElement(lp, 'difficult')
			difficult.text = '0'

			# Bounding Box
			bndbox = ET.SubElement(lp, 'bndbox')
			xmin = ET.SubElement(bndbox, 'xmin')
			xmin.text = str(bbox.xmin)
			ymin = ET.SubElement(bndbox, 'ymin')
			ymin.text = str(bbox.ymin)
			xmax = ET.SubElement(bndbox, 'xmax')
			xmax.text = str(bbox.xmax)
			ymax = ET.SubElement(bndbox, 'ymax')
			ymax.text = str(bbox.ymax)
			
			# Text
			plate = ET.SubElement(bndbox, 'plate')
			plate.text = format_plate(txt)
		
		root_indent(root)
		tree = ET.ElementTree(root)
	
	if not predictions:
		tree = root

	return tree

def write_tree(tree, output_path):
	tree.write(output_path)

def root_indent(elem, level=0):
	i = "\n" + level * "    "
	if len(elem):
		if not elem.text or not elem.text.strip():
			elem.text = i + "    "
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
		for elem in elem:
			root_indent(elem, level+1)
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
	else:
		if level and (not elem.tail or not elem.tail.strip()):
			elem.tail = i
	return None
	
def format_plate(plate):
	lp = ''.join(plate)
	return lp.upper()

if __name__ == '__main__':
	pass
	# YOLO predictions	[[bbox, bbox, bbox],img,...,[bbox, bbox]]
	# self.xmin = xmin
	# self.ymin = ymin
	# self.xmax = xmax
	# self.ymax = ymax
	# LSTM predictions [ [['aa', '205', 'nx'], ['gdt', '465']] ,img,...,img]