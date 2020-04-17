import sys, os
import keras
import cv2
import traceback
import regex
from PIL import Image
import numpy as np

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename, isfile, isdir
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes, lread, Label, readShapes
from math						import sqrt

def dist(pt0, pt1):  
	return sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':

	try:
		
		input_dir  = sys.argv[1] + '_tmp'
		output_dir = input_dir
		source_dir  = sys.argv[1]

		lp_threshold = .5

		wpod_net_path = sys.argv[2]
		wpod_net = load_model(wpod_net_path)

		imgs_paths = glob('%s/*car.png' % input_dir)

		print 'Searching for license plates using WPOD-NET'

		for i,img_path in enumerate(imgs_paths):

			print '\t Processing %s' % img_path

			bname = splitext(basename(img_path))[0]
			Ivehicle = cv2.imread(img_path)

			ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
			side  = int(ratio*288.)
			bound_dim = min(side + (side%(2**4)),608)
			print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)

			# Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
			Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,160),lp_threshold)

			if len(LlpImgs):
				Ilp = LlpImgs[0]

				cv2.imwrite('%s/%s_lpc.png' % (output_dir,bname),Ilp*255.)

				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

				s = Shape(Llp[0].pts)

				source_file_name, icar = regex.search('(\S+)_(\d)+car', bname).groups()
				full_path = glob('%s/%s.*' % (source_dir, source_file_name))[0]
				source_img = Image.open(full_path)
				print source_img.size

				detected_cars_labels = '%s/%s_cars.txt' % (output_dir,source_file_name)
				Lcar = lread(detected_cars_labels)
				lcar = Lcar[int(icar)]
				pts = s.pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
				ptspx = pts*np.array(source_img.size, dtype=float).reshape(2,1)
				print ptspx

				pt0 = (ptspx[0,0], ptspx[1,0])
				pt1 = (ptspx[0,1], ptspx[1,1])
				pt2 = (ptspx[0,2], ptspx[1,2])
				width = dist(pt0, pt1)
				height = dist(pt1, pt2)
				ratio = width / height
				print width
				print height
				print ratio
				if ratio < 2:
					print "! Two Lined !"
					s.two_lined = True

				cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
				writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
			
			else:
				print "!! License Plate NOT Found !!"

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)


