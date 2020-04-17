import sys
import cv2
import numpy as np
import traceback

from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder


if __name__ == '__main__':
    
    try:

        input_dir  = sys.argv[1]
        output_dir = sys.argv[2]

        imgs_paths = image_files_from_folder(input_dir)
        imgs_paths.sort()

        if not isdir(output_dir):
            makedirs(output_dir)

        for i,img_path in enumerate(imgs_paths):
            img = cv2.imread(img_path)
            bname = basename(splitext(img_path)[0])

            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            # equalize the histogram of the Y channel
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            # convert the YUV image back to RGB format
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            cv2.imwrite('%s/%s.jpg' % (output_dir,bname),img_output)

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
