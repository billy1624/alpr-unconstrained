import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder
from darknet.python.darknet import detect


if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = sys.argv[1] + '_tmp'

		vehicle_threshold = .5

		vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = 'data/vehicle-detector/voc.data'

		vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
		vehicle_meta = dn.load_meta(vehicle_dataset)

		imgs_paths = image_files_from_folder(input_dir)
		imgs_paths.sort()

		if not isdir(output_dir):
			makedirs(output_dir)

		print 'Searching for vehicles using YOLO...'

		for i,img_path in enumerate(imgs_paths):

			print '\tScanning %s' % img_path

			bname = basename(splitext(img_path)[0])

			R,_ = detect(vehicle_net, vehicle_meta, img_path ,thresh=vehicle_threshold)

			R = [r for r in R if r[0] in ['car','bus']]

			print '\t\t%d cars found' % len(R)

			if len(R):

				Iorig = cv2.imread(img_path)
				WH = np.array(Iorig.shape[1::-1],dtype=float)
				Lcars = []

				for i,r in enumerate(R):

					cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
					tl = np.array([cx - w/2., cy - h/2.])
					br = np.array([cx + w/2., cy + h/2.])

					pos = 'N/A'
					center_x = (br[0] + tl[0]) / 2
					height = br[1] - tl[1]

					if height >= 0.2:
						if center_x < 0.25:
							pos = 'left'
						elif center_x >= 0.25 and center_x <= 0.75:
							pos = 'middle'
						elif center_x > 0.75:
							pos = 'right'

					label = Label(0,tl,br,pos=pos)

					print("{}, center_x={}, height={}, pos={}".format(label, center_x, height, pos))

					Lcars.append(label)

					if pos == 'middle':
						Icar = crop_region(Iorig,label)
						cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,i),Icar)
				
				print('')

				lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)
			
			# else:

			# 	Iorig = cv2.imread(img_path)
			# 	WH = np.array(Iorig.shape[1::-1],dtype=float)
			# 	Lcars = []

			# 	cx,cy,w,h = (np.array((0,0,WH[0],WH[1])) / np.concatenate( (WH,WH) )).tolist()
			# 	tl = np.array([cx, cy])
			# 	br = np.array([cx+w, cy+h])
			# 	label = Label(0,tl,br)
			# 	Icar = crop_region(Iorig,label)

			# 	Lcars.append(label)

			# 	cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,0),Icar)

			# 	lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
	