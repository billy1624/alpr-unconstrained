import sys
import cv2
import numpy as np
import traceback

from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder, image_files_from_api

from datetime import datetime
from datetime import timedelta


if __name__ == '__main__':
    
    try:

        input_dir           = sys.argv[1]
        output_dir          = sys.argv[2]
        is_api              = sys.argv[3]
        keep_original_image = sys.argv[4]
        sampling            = sys.argv[5]

        if not isdir(output_dir):
            makedirs(output_dir)

        if is_api == '1':
            imgs_paths = image_files_from_api(input_dir)
        else:
            imgs_paths = image_files_from_folder(input_dir)


        if sampling == '1':

            if not isdir(output_dir + '_ori'):
                makedirs(output_dir + '_ori')

            imgs_paths_time = {}

            for i,img_path in enumerate(imgs_paths):
                time = basename(splitext(img_path)[0]).split('_')[1]
                time = time[:6]
                imgs_paths_time[time] = img_path
                
            # print 'files in "{}"\n'.format(input_dir)
            # for k in sorted(imgs_paths_time.keys()):
            #     print k, imgs_paths_time[k]
            # print "\n"

            new_imgs_paths = {}

            time_format = '%H%M%S'
            start_hour = 7
            mins = 60

            time_start = datetime.now().replace(hour=start_hour, minute=0, second=0, microsecond=0)
            time_end = datetime.now().replace(hour=start_hour, minute=0, second=0, microsecond=0)
            time_end = time_start + timedelta(minutes=mins)

            time_start_int = int(time_start.strftime(time_format))
            time_end_int = int(time_end.strftime(time_format))

            while time_start_int <= 210000:
                new_imgs_paths[time_start_int] = None
                for k in sorted(imgs_paths_time.keys()):
                    if int(k) >= time_start_int and int(k) <= time_end_int:
                        new_imgs_paths[time_start_int] = imgs_paths_time[k]
                        break
                
                time_start = time_start + timedelta(minutes=mins)
                time_end = time_start + timedelta(minutes=mins)
                time_start_int = int(time_start.strftime(time_format))
                time_end_int = int(time_end.strftime(time_format))

            # print 'Selected images:\n'
            # for k in sorted(new_imgs_paths.keys()):
            #     print k, new_imgs_paths[k]
            # print "\n"

            imgs_paths = []

            for k in sorted(new_imgs_paths.keys()):
                if new_imgs_paths[k] is not None:
                    imgs_paths.append(new_imgs_paths[k])
            
            print 'New imgs_paths:\n'
            for path in imgs_paths:
                print path
            print "\n"


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
            
            if sampling == '1':
                cv2.imwrite('%s_ori/%s%s.jpg' % (output_dir,id_str,bname),img)

            if is_api == '1' or keep_original_image == '0':
                cv2.imwrite('%s/%s%s.jpg' % (output_dir,id_str,bname),img_output)
            else:
                cv2.imwrite('%s/%s%s.jpg' % (output_dir,id_str,bname),img)
                cv2.imwrite('%s/%s%s_lighten.jpg' % (output_dir,id_str,bname),img_output)

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
