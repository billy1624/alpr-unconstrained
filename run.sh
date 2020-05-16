#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

check_file() 
{
	if [ ! -f "$1" ]
	then
		return 0
	else
		return 1
	fi
}

check_dir() 
{
	if [ ! -d "$1" ]
	then
		return 0
	else
		return 1
	fi
}


# Check if Darknet is compiled
check_file "darknet/libdarknet.so"
retval=$?
if [ $retval -eq 0 ]
then
	echo "Darknet is not compiled! Go to 'darknet' directory and 'make'!"
	exit 1
fi

lp_model="data/lp-detector/wpod-net_update1.h5"
input_dir=''
output_dir=''
csv_file=''
is_api=0
keep_original_image=0
sampling=0


# Check # of arguments
usage() {
	echo ""
	echo " Usage:"
	echo ""
	echo "   bash $0 -i input/dir -o output/dir -c csv_file.csv [-h] [-l path/to/model]:"
	echo ""
	echo "   -i   Input dir path (containing JPG or PNG images)"
	echo "   -o   Output dir path"
	echo "   -c   Output CSV file path"
	echo "   -l   Path to Keras LP detector model (default = $lp_model)"
	echo "   -h   Print this help information"
	echo ""
	exit 1
}

while getopts 'i:o:c:l:k:s:h' OPTION; do
	case $OPTION in
		i) input_dir=$OPTARG;;
		o) output_dir=$OPTARG;;
		c) csv_file=$OPTARG;;
		l) lp_model=$OPTARG;;
		k) keep_original_image=$OPTARG;;
		s) sampling=$OPTARG;;
		h) usage;;
	esac
done

date_time=$(date +'%Y%m%d_%H%M%S')

if [ -z "$input_dir"  ]; then echo "Input dir not set."; usage; exit 1; fi
if [ -z "$output_dir" ]; then output_dir="/tmp/alpr_$date_time"; is_api=1; fi
if [ -z "$keep_original_image" ]; then keep_original_image=0; fi
if [ -z "$sampling" ]; then sampling=0; fi

# # Check if input dir exists
# check_dir $input_dir
# retval=$?
# if [ $retval -eq 0 ]
# then
# 	echo "Input directory ($input_dir) does not exist"
# 	exit 1
# fi

# # Check if output dir exists, if not, create it
# check_dir $output_dir
# retval=$?
# if [ $retval -eq 0 ]
# then
# 	mkdir -p $output_dir
# fi

# End if any error occur
set -e

# Preprocess images
python image-preprocessing.py $input_dir $output_dir $is_api $keep_original_image $sampling

# Detect vehicles
python vehicle-detection.py $output_dir

# Detect license plates
python license-plate-detection.py $output_dir $lp_model

# OCR
python license-plate-ocr.py $output_dir

# Draw output and generate list
python gen-outputs.py $output_dir $is_api

# Clean files and draw output
if [ $is_api -eq 1 ]
then
	echo ""
	echo "Removing tmp files..."
	echo "rm -r ${output_dir}"
	rm -r ${output_dir}
	echo "rm -r ${output_dir}_tmp"
	rm -r ${output_dir}_tmp
	echo "rm -r ${output_dir}_out"
	rm -r ${output_dir}_out
fi

# rm $output_dir/*_lp.png
# rm $output_dir/*car.png
# rm $output_dir/*_cars.txt
# rm $output_dir/*_lp.txt
# rm $output_dir/*_str.txt
