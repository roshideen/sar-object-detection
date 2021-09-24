import time
from snappy import ProductIO,HashMap,GPF
import os,gc
import snappy
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson
import glob
import subprocess
import jpy
from zipfile import ZipFile
from IPython.display import display
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import h5py
import cropped_sar_preparing
import gc
from datetime import datetime
import argparse


def parse_opt(known=False):
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_group',type=str)
	parser.add_argument('--sar_filename',type=str)
	parser.add_argument('--path_product', type=str )
	parser.add_argument('--path_saved_img', type=str )
	parser.add_argument('--path_saved_label', type=str )
	parser.add_argument('--path_saved_sample', type=str )
	parser.add_argument('--path_saved_full_sample', type=str )
	parser.add_argument('--path_saved_nolabel', type=str )
	parser.add_argument('--ksize',type=int)
	parser.add_argument('--overlap_split',type=bool)
	parser.add_argument('--calibrate_sar',type=bool,default=False)
	opt = parser.parse_known_args()[0] if known else parser.parse_args()
	return opt

def main(opt):
	dataset_group = opt.dataset_group
	product_path = os.path.join(opt.path_product,dataset_group)
	path_saved_img = os.path.join(opt.path_saved_img,dataset_group)	
	path_saved_label = 	os.path.join(opt.path_saved_label,dataset_group)
	path_saved_sample = os.path.join(opt.path_saved_sample,dataset_group)
	path_saved_full_sample = os.path.join(opt.path_saved_full_sample,dataset_group)
	path_saved_nolabel = os.path.join(opt.path_saved_nolabel,dataset_group)
	ksize = opt.ksize
	sar_filename = opt.sar_filename	
	overlap_split = opt.overlap_split



	kanum_path = r'D:\08 aimagin\satellite image project\sat\coordinate_khanum'
	kanum_file = 'kanum_map'+'.csv'
	df_kanum_ori = pd.read_csv(os.path.join(kanum_path,kanum_file))
	df_kanum = df_kanum_ori.copy()

	mask_path = r'D:\08 aimagin\satellite image project\sar'
	gulf_mask = cv2.imread(os.path.join(mask_path,'gulf_mask_SAR2.png'),-1)
	gulf_mask[gulf_mask==255]=1

	all_paths = [path_saved_img, path_saved_label,path_saved_label,path_saved_sample, path_saved_full_sample, path_saved_nolabel]
	
	day_sar_filename = sar_filename.split('\\')[-1].split('_')[4].split('T')[0]
	print(day_sar_filename)
	s1_read = snappy.ProductIO.readProduct(sar_filename)
	subset,band_nm = cropped_sar_preparing.get_sar_product_subset(s1_read,calibrate_sar=opt.calibrate_sar)
	band_all_data = cropped_sar_preparing.get_all_masked_band_GRDSAR(subset,gulf_mask,calibrate_sar=opt.calibrate_sar)
	if not opt.calibrate_sar:
		tmp = band_all_data[3].copy() # select only Intensity_VV to plot full sample
	else:
		tmp = band_all_data[0].copy()
	if not overlap_split:
		
		band_all_data = cropped_sar_preparing.get_split_bands(band_all_data,ksize)

		cropped_sar_preparing.tiling_saving_images(subset,band_all_data,tmp,df_kanum,gulf_mask,day_sar_filename,all_paths,ksize)
	else:
		band_all_data,x_pts,y_pts = cropped_sar_preparing.get_overlapping_split(band_all_data,ksize)
		points = (x_pts,y_pts)
		cropped_sar_preparing.tiling_saving_images_overlapped(subset,band_all_data,points,tmp,df_kanum,gulf_mask,day_sar_filename,all_paths,ksize,calibrate_sar=opt.calibrate_sar)
	print('\tcompleted!')
	del s1_read,subset,band_all_data,tmp
	gc.collect()



if __name__ == "__main__":
	opt = parse_opt()
	main(opt)