import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from os.path 				import splitext, basename
from glob					import glob
from darknet.python.darknet import detect
from src.label				import dknet_label_conversion
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

			R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)

			if len(R):

				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)

				L.sort(key=lambda x: x.tl()[0])
				lp_str = ''.join([chr(l.cl()) for l in L])

				with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					f.write(lp_str + '\n')

				print '\t\tLP: %s' % lp_str

			else:

				print 'No characters found'

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
