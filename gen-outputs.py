import sys
import cv2
import numpy as np

from glob						import glob
from os.path 					import splitext, basename, isfile, isdir
from os 						import makedirs
from src.utils 					import crop_region, image_files_from_folder
from src.drawing_utils			import draw_label, draw_losangle, write2img
from src.label 					import lread, Label, readShapes
from math						import sqrt

from pdb import set_trace as pause


def dist(pt0, pt1):  
	return sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)


YELLOW = (  0,255,255)
RED    = (  0,  0,255)

input_dir = sys.argv[1]
tmp_dir = sys.argv[1] + '_tmp'
output_dir = sys.argv[1] + '_out'

if not isdir(output_dir):
	makedirs(output_dir)

img_files = image_files_from_folder(input_dir)

for img_file in img_files:

	bname = splitext(basename(img_file))[0]

	I = cv2.imread(img_file)

	detected_cars_labels = '%s/%s_cars.txt' % (tmp_dir,bname)

	Lcar = lread(detected_cars_labels)

	# sys.stdout.write('%s' % bname)

	if Lcar:

		for i,lcar in enumerate(Lcar):

			draw_label(I,lcar,color=YELLOW,thickness=3)

			lp_label 		= '%s/%s_%dcar_lp.txt'		% (tmp_dir,bname,i)
			lp_label_str 	= '%s/%s_%dcar_lp_str.txt'	% (tmp_dir,bname,i)

			if isfile(lp_label):

				Llp_shapes = readShapes(lp_label)
				pts = Llp_shapes[0].pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
				ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
				draw_losangle(I,ptspx,RED,3)

				# print ""
				# print ""
				# print lp_label

				# print I.shape
				# print I.shape[1::-1]

				# pt0 = (ptspx[0,0], ptspx[1,0])
				# pt1 = (ptspx[0,1], ptspx[1,1])
				# pt2 = (ptspx[0,2], ptspx[1,2])
				# width = dist(pt0, pt1)
				# height = dist(pt1, pt2)
				# ratio = width / height
				# print width
				# print height
				# print ratio
				# if ratio < 2:
				# 	print "Two Lined!"

				if isfile(lp_label_str):
					with open(lp_label_str,'r') as f:
						lp_str = f.read().strip()
					llp = Label(0,tl=pts.min(1),br=pts.max(1))
					write2img(I,llp,lp_str)

					# sys.stdout.write(',%s' % lp_str)

	cv2.imwrite('%s/%s_output.png' % (output_dir,bname),I)
	# sys.stdout.write('\n')


