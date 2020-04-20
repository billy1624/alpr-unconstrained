import sys
import cv2
import numpy as np
import traceback

from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder, image_files_from_api


if __name__ == '__main__':
    
    try:

        input_dir  = sys.argv[1]
        output_dir = sys.argv[2]
        is_api     = sys.argv[3]

        if not isdir(output_dir):
            makedirs(output_dir)

        if is_api == '1':
            imgs_paths = image_files_from_api(input_dir)
        else:
            imgs_paths = image_files_from_folder(input_dir)

        if not isdir(output_dir):
            makedirs(output_dir)

        for i,img_path in enumerate(imgs_paths):

            parking_space_id = None
            if is_api == '1':
                parking_space_id = img_path[1]
                img_path = img_path[0]

            img = cv2.imread(img_path)
            bname = basename(splitext(img_path)[0])

            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            # equalize the histogram of the Y channel
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            # convert the YUV image back to RGB format
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            
            id_str = ''
            if is_api == '1':
                id_str = 'ID' + parking_space_id + '_'

            # cv2.imwrite('%s/%s%s.jpg' % (output_dir,id_str,bname),img)
            # cv2.imwrite('%s/%s%s_lighten.jpg' % (output_dir,id_str,bname),img_output)

            cv2.imwrite('%s/%s%s.jpg' % (output_dir,id_str,bname),img_output)

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
