import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from os.path 				import splitext, basename, isfile, isdir
from glob					import glob
from darknet.python.darknet import detect
from src.label				import dknet_label_conversion, lread, Label, readShapes
from src.utils 				import nms


if __name__ == '__main__':

	try:

		input_dir  = sys.argv[1] + '_tmp'
		output_dir = input_dir

		ocr_threshold = .4

		ocr_weights = 'data/ocr/ocr-net.weights'
		ocr_netcfg  = 'data/ocr/ocr-net.cfg'
		ocr_dataset = 'data/ocr/ocr-net.data'

		ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)

		imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

		print 'Performing OCR...'

		for i,img_path in enumerate(imgs_paths):

			# /tmp/output_img_20200301_mix_tmp/20200114_221658_0car_lp.txt
			# [[0.30671  0.696869 0.706321 0.316162]
			#  [0.574803 0.526937 0.659712 0.707579]]
			# [[696.7229943  816.42938529 815.98953454 696.28314355]
			#  [718.95660457 718.55229915 750.58574764 750.99005306]]
			# 0 (696, 718) (816, 718)
			# 1 (816, 718) (815, 750)
			# 2 (815, 750) (696, 750)
			# 3 (696, 750) (696, 718)

			print '\tScanning %s' % img_path

			bname = basename(splitext(img_path)[0])

			# source_file_name = regex.search('(\S+)_\d+car_lp', bname).groups()[0]
			# full_path = glob('%s/%s.*' % (source_dir, source_file_name))[0]
			# source_img = Image.open(full_path)
			# print source_img.size

			lp_str = ''

			lp_label = '%s/%s.txt' % (output_dir, bname)
			Llp_shapes = readShapes(lp_label)[0]

			# if Llp_shapes.two_lined:
			if False:
				print '! Two Lined !'

				img = cv2.imread(img_path)

				top_img = cv2.rectangle(img.copy(), (0,45), (240,80), (255,255,255), -1)
				top_img_path = '%s/%s_1.png' % (output_dir, bname)
				cv2.imwrite(top_img_path, top_img)

				R,(width,height) = detect(ocr_net, ocr_meta, top_img_path ,thresh=ocr_threshold, nms=None)
				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)
				L.sort(key=lambda x: x.tl()[0])
				lp_str += ''.join([chr(l.cl()) for l in L])

				bottom_img = cv2.rectangle(img.copy(), (0,0), (240,35), (255,255,255), -1)
				bottom_img_path = '%s/%s_2.png' % (output_dir, bname)
				cv2.imwrite(bottom_img_path, bottom_img)

				R,(width,height) = detect(ocr_net, ocr_meta, bottom_img_path ,thresh=ocr_threshold, nms=None)
				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)
				L.sort(key=lambda x: x.tl()[0])
				lp_str += ''.join([chr(l.cl()) for l in L])
			
			else:
				R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)
				# print("R: {}\n".format(R))
				L = dknet_label_conversion(R,width,height)
				# print("dknet_label_conversion(R,width,height): {}\n".format(L))
				L = nms(L,.45)
				# print("nms(L,.45): {}\n".format(L))
				L.sort(key=lambda x: x.tl()[0])
				# print("L.sort(key=lambda x: x.tl()[0]): {}\n".format(L))
				print("L:")
				if len(L):
					avg = []
					for l in L:
						print("{} {} wh={}\n".format(chr(l.cl()), l, l.wh()))
						avg.append(l.wh())
					avg = np.array(avg)
					avg = np.average(avg, axis=0)
					minimum_tl_y = avg[1] / 2
					print(avg)
					print(minimum_tl_y)
					print_L = []
					for l in L:
						if l.tl()[1] <= minimum_tl_y:
							lp_str += chr(l.cl())
						else:
							print_L.append(l)
					lp_str += ''.join([chr(l.cl()) for l in print_L])

			if len(lp_str):
				with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					f.write(lp_str + '\n')
				print '\t\tLP: %s' % lp_str

			else:
				print 'No characters found'

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
