# import datetime
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
from datetime import datetime

def reshape_split(image0:np.ndarray,kernel_size:int):
	image = image0.copy()
	img_height,img_width = image.shape[0:2]
	lim_height = (img_height//kernel_size)*kernel_size
	lim_width = (img_width//kernel_size)*kernel_size
	image = image[:lim_height,:lim_width]
	
	img_height,img_width = image.shape[0:2]
	tile_height,tile_width = kernel_size,kernel_size
	tiled_array = image.reshape(img_height//tile_height,tile_height,img_width//tile_width,tile_width)
	tiled_array = tiled_array.swapaxes(1,2)
	del image0,image
	return tiled_array

def get_sar_product_subset(s1_read,calibrate_sar:bool=False,lat:float=9.126140086498603,long:float=99.20271266495655,width:int=5000,height:int=3000):
	x,y = utils.convert_latlong2pixel_sentinel1(s1_read,lat,long)
	x,y = int(x),int(y)
	if calibrate_sar:
		s1_read = utils.do_apply_orbit_file(s1_read)
		polarization, pols = 'DV','VV,VH'
		s1_read = utils.do_calibration(s1_read, polarization, pols)
	subset = utils.get_subset_product(s1_read,x,y,width,height)
	band_nm = utils.get_band_names(subset)
	subset = utils.do_terrain_correction(subset,band_nm, downsample=False)
	del s1_read
	return subset,band_nm

def get_all_masked_band_GRDSAR(subset,gulf_mask,calibrate_sar:bool=False):
	if not calibrate_sar:
		band1 = utils.project2img(subset,'Amplitude_VH') #int16
		band2 = utils.project2img(subset,'Intensity_VH') #int32
		band3 = utils.project2img(subset,'Amplitude_VV') #int16
		band4 = utils.project2img(subset,'Intensity_VV') #int32
		numerator = (band2*band4)
		denominator = (band1*band4)+(band2*band3)
		band5 = np.where(denominator==0,0,numerator/denominator)
		band_all_data = [band1,band2,band3,band4,band5]
		del band1, band2, band3, band4, band5,numerator, denominator
	else:
		band1 = utils.project2img(subset,list(subset.getBandNames())[0]) #int16
		band2 = utils.project2img(subset,list(subset.getBandNames())[1]) #int32
		band3 = utils.project2img(subset,list(subset.getBandNames())[2]) #int16
		band_all_data = [band1,band2,band3]
		del band1, band2, band3
		
	xlim = np.min([band_all_data[0].shape[1],gulf_mask.shape[1]])
	ylim = np.min([band_all_data[0].shape[0],gulf_mask.shape[0]])
	band_all_data = [data_i[:ylim,:xlim]*gulf_mask[:ylim,:xlim] for data_i in band_all_data]
	
	return band_all_data

def get_split_bands(band_all_data,ksize):
	return [reshape_split(data_i,ksize) for data_i in band_all_data]

def tiling_saving_images(subset,split_band_all_data,full_img,df_kanumx,gulf_maskx,day_sar_filename,saved_paths,ksize,fontScale:int=3,thickness:int=10,color:int=5e3):
	df_kanum=df_kanumx.copy()
	gulf_mask = gulf_maskx.copy()
	tmp = full_img.copy()
	band_all_data = split_band_all_data.copy()
	path_saved_img, path_saved_label,path_saved_label,path_saved_sample, path_saved__full_sample, path_saved_nolabel = saved_paths
	



	column_kanum_tobekept = [i for i in df_kanum.columns if 'kanumFoundDate' in i]
	df_kanum.rename(columns={df_kanum.columns[0]:'kanum_id'},inplace=True)
	df_kanum.set_index('kanum_id',inplace=True)

	fmt = '%Y%m%d'
	date1 = datetime.strptime(day_sar_filename,fmt)
	datecolumn_fmt = [datetime.strptime(x.split('_')[1],fmt) for x in column_kanum_tobekept]
	gtdate_column = column_kanum_tobekept[(np.where((np.array([(date1-x).days for x in datecolumn_fmt])<=0))[0][0])]
	found_kanums = df_kanum.index[np.where(df_kanum.loc[:,gtdate_column].notnull())[0]]
	found_kanums_pixel = [df_kanum.loc[list(found_kanums),['latitude','longitude']].apply(lambda x: utils.convert_latlong2pixel_sentinel1(subset,*x),axis=1)]
	found_kanums_pixel = list(found_kanums_pixel[0].values)
	cropped_all_bands = np.zeros((5,ksize,ksize))

	# all constant param is configed for Intensity VV only
	# %matplotlib inline
	tiled_width, tiled_height = ksize, ksize
	img_full_sample = tmp.copy()
	xmsk_white_min, xmsk_white_max = np.where(gulf_mask==1)[1].min(),np.where(gulf_mask==1)[1].max()
	ymsk_white_min, ymsk_white_max = np.where(gulf_mask==1)[0].min(),np.where(gulf_mask==1)[0].max()

	savedtiled_img_count = 0
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	box_width,box_height = 5,5
	for i in range(0,band_all_data[3].shape[0]):
		for j in range(0,band_all_data[3].shape[1]):
			y1,y2 = i*ksize,(i+1)*ksize
			x1,x2 = j*ksize,(j+1)*ksize
			#saved only images overlapped in masked area
			if ((x2>xmsk_white_min) and (y2>ymsk_white_min)) or ((xmsk_white_min<x1<xmsk_white_max) and (y2>ymsk_white_min)):
				
				fl_name_saved = day_sar_filename+'_'+str(savedtiled_img_count)+'_x0_'+str(x1)+'_y0_'+str(y1)+'_ksize_'+str(ksize)
				sample = band_all_data[3][i,j,:,:].copy()
				kanum_in_cropped_pixel = [k for k in found_kanums_pixel if (x1<k[0]<x2) and (y1<k[1]<y2) ]
				tiled_img = band_all_data[3][i,j,:,:].copy()
				if (tiled_img ==0).all():
					continue
				# confirmed saving if image not all zero!
				savedtiled_img_count+=1
				cropped_all_bands = np.zeros((5,ksize,ksize)) ## >>> cropped bands to be saved
				for band_idx in range(0,5):
					cropped_all_bands[band_idx,:,:] = band_all_data[band_idx][i,j,:,:].copy()
				
				if len(kanum_in_cropped_pixel) != 0:
					df_label = pd.DataFrame(columns=[1,2,3,4,5]) ## >>> label df to be saved
					for nrow,k in enumerate(kanum_in_cropped_pixel):
						xc,yc = int(k[0]),int(k[1])
						xcn,ycn = int(k[0]-x1),int(k[1]-y1)
						sample = cv2.circle(sample,(xcn,ycn),8,0,2)
						img_full_sample = cv2.circle(img_full_sample,(xc,yc),8,0,2)
						row_data = [0,xcn/ksize,ycn/ksize,box_width/ksize,box_height/ksize]
						df_label.loc[len(df_label)]=row_data
					###########saving labels############
					file_saved_label = os.path.join(path_saved_label,fl_name_saved+'.txt')
					df_label.to_csv(os.path.join(file_saved_label),sep=' ',index=False,header=False)
					###########saving labels############
					file_saved_cropped_bands = os.path.join(path_saved_img,fl_name_saved+'.npy')
				else:
					file_saved_cropped_bands = os.path.join(path_saved_nolabel,fl_name_saved+'.npy')

				###########saving bands############
				
				saved_bands_data_npy =  h5py.File(file_saved_cropped_bands, "w")
				saved_bands_data_npy.create_dataset("sar", np.shape(cropped_all_bands), dtype=np.float64, data=cropped_all_bands)
				###########saving bands############
				txt = str(savedtiled_img_count)
				org = (x1,y1)
				img_full_sample = cv2.rectangle(img_full_sample,(x1,y1),(x2,y2),5e3,5)
				img_full_sample = cv2.putText(img_full_sample, txt, org, font,fontScale, color, thickness, cv2.LINE_AA)
	#                 break_or_not=True
				
				###########saving croppeds-samples Intensity VV############
				file_saved_sample = os.path.join(path_saved_sample,fl_name_saved+'.png')
				fig,ax = plt.subplots()
				ax.imshow(sample,cmap='gray',vmin=300,vmax=1e4)
				fig.savefig(file_saved_sample)
				fig.clear()
				plt.close()
				###########saving croppeds-samples Intensity VV############
	###########saving full-sample Intensity VV############
	file_saved_full_sample = os.path.join(path_saved__full_sample,day_sar_filename+'_full.png')
	fig,ax = plt.subplots(figsize=(20,10))
	ax.imshow(img_full_sample,cmap='gray',vmin=300,vmax=1e4)
	fig.savefig(file_saved_full_sample,dpi=300)
	fig.clear()
	plt.close()
	###########saving full-sample Intensity VV############

def reshape_overlapping(kernel_size,img0):
    img = img0.copy()
    img_height, img_width = img.shape[0:2]
    lim_height = (img_height//kernel_size)*kernel_size
    lim_width = (img_width//kernel_size)*kernel_size

    img = img[:lim_height,:lim_width]
    img_height, img_width = img.shape[0:2]
    X_points = start_points(img_width, kernel_size, 0.5)
    Y_points = start_points(img_height, kernel_size, 0.5)
    tiles = np.zeros((len(Y_points),len(X_points),kernel_size,kernel_size))
    for a,i in enumerate(Y_points):    
        for b,j in enumerate(X_points):
            tiles[a,b,:,:] = img[i:i+kernel_size, j:j+kernel_size]
    return tiles, X_points, Y_points

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def get_overlapping_split(band_all,kernel_size):
    overlapped_splits = []
    for i,bandx in enumerate(band_all):
        if i == 0:
            tiles, X_points, Y_points = reshape_overlapping(kernel_size,bandx)
            overlapped_splits.append(tiles)
        else:
            tiles, _, _ = reshape_overlapping(kernel_size,bandx)
            overlapped_splits.append(tiles)
        
    return overlapped_splits,X_points,Y_points

def tiling_saving_images_overlapped(subset,split_band_all_data,points,full_img,df_kanumx,gulf_maskx,day_sar_filename,saved_paths,ksize,fontScale:int=3,thickness:int=10,color:int=5e3,calibrate_sar:bool=False):
	df_kanum=df_kanumx.copy()
	gulf_mask = gulf_maskx.copy()
	tmp = full_img.copy()
	band_all_data = split_band_all_data.copy()
	path_saved_img, path_saved_label,path_saved_label,path_saved_sample, path_saved__full_sample, path_saved_nolabel = saved_paths
	



	column_kanum_tobekept = [i for i in df_kanum.columns if 'kanumFoundDate' in i]
	df_kanum.rename(columns={df_kanum.columns[0]:'kanum_id'},inplace=True)
	df_kanum.set_index('kanum_id',inplace=True)

	fmt = '%Y%m%d'
	date1 = datetime.strptime(day_sar_filename,fmt)
	datecolumn_fmt = [datetime.strptime(x.split('_')[1],fmt) for x in column_kanum_tobekept]
	gtdate_column = column_kanum_tobekept[(np.where((np.array([(date1-x).days for x in datecolumn_fmt])<=0))[0][0])]
	found_kanums = df_kanum.index[np.where(df_kanum.loc[:,gtdate_column].notnull())[0]]
	found_kanums_pixel = [df_kanum.loc[list(found_kanums),['latitude','longitude']].apply(lambda x: utils.convert_latlong2pixel_sentinel1(subset,*x),axis=1)]
	found_kanums_pixel = list(found_kanums_pixel[0].values)
	cropped_all_bands = np.zeros((5,ksize,ksize))

	# all constant param is configed for Intensity VV only
	# %matplotlib inline
	tiled_width, tiled_height = ksize, ksize
	img_full_sample = tmp.copy()
	xmsk_white_min, xmsk_white_max = np.where(gulf_mask==1)[1].min(),np.where(gulf_mask==1)[1].max()
	ymsk_white_min, ymsk_white_max = np.where(gulf_mask==1)[0].min(),np.where(gulf_mask==1)[0].max()

	savedtiled_img_count = 0
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	box_width,box_height = 5,5
	X_points, Y_points = points
	for i,a in enumerate(Y_points):
		for j,b in enumerate(X_points):
			y1,y2 = a,a+ksize
			x1,x2 = b,b+ksize
			#saved only images overlapped in masked area
			if ((x2>xmsk_white_min) and (y2>ymsk_white_min)) or ((xmsk_white_min<x1<xmsk_white_max) and (y2>ymsk_white_min)):
				
				fl_name_saved = day_sar_filename+'_'+str(savedtiled_img_count)+'_x0_'+str(x1)+'_y0_'+str(y1)+'_ksize_'+str(ksize)
				if not calibrate_sar:
					channel_plot = 3
					vmin,vmax = 300,1e4
					total_channel = 5
				else:
					channel_plot = 0
					vmin,vmax = 0.01, 0.1
					color = 0.5
					total_channel = 3

				sample = band_all_data[channel_plot][i,j,:,:].copy()
				kanum_in_cropped_pixel = [k for k in found_kanums_pixel if (x1<k[0]<x2) and (y1<k[1]<y2) ]
				tiled_img = band_all_data[channel_plot][i,j,:,:].copy()
				if (tiled_img ==0).all():
					continue
				# confirmed saving if image not all zero!
				savedtiled_img_count+=1
				cropped_all_bands = np.zeros((total_channel,ksize,ksize)) ## >>> cropped bands to be saved
				for band_idx in range(0,total_channel):
					cropped_all_bands[band_idx,:,:] = band_all_data[band_idx][i,j,:,:].copy()
				
				if len(kanum_in_cropped_pixel) != 0:
					df_label = pd.DataFrame(columns=[1,2,3,4,5]) ## >>> label df to be saved
					for nrow,k in enumerate(kanum_in_cropped_pixel):
						xc,yc = int(k[0]),int(k[1])
						xcn,ycn = int(k[0]-x1),int(k[1]-y1)
						sample = cv2.circle(sample,(xcn,ycn),8,0,2)
						img_full_sample = cv2.circle(img_full_sample,(xc,yc),8,0,2)
						row_data = [0,xcn/ksize,ycn/ksize,box_width/ksize,box_height/ksize]
						df_label.loc[len(df_label)]=row_data
					###########saving labels############
					file_saved_label = os.path.join(path_saved_label,fl_name_saved+'.txt')
					df_label.to_csv(os.path.join(file_saved_label),sep=' ',index=False,header=False)
					###########saving labels############
					file_saved_cropped_bands = os.path.join(path_saved_img,fl_name_saved+'.npy')
				else:
					file_saved_cropped_bands = os.path.join(path_saved_nolabel,fl_name_saved+'.npy')

				###########saving bands############
				
				saved_bands_data_npy =  h5py.File(file_saved_cropped_bands, "w")
				saved_bands_data_npy.create_dataset("sar", np.shape(cropped_all_bands), dtype=np.float64, data=cropped_all_bands)
				###########saving bands############
				txt = str(savedtiled_img_count)
				org = (x1,y1)
				img_full_sample = cv2.rectangle(img_full_sample,(x1,y1),(x2,y2),5e3,5)
				img_full_sample = cv2.putText(img_full_sample, txt, org, font,fontScale, color, thickness, cv2.LINE_AA)
	#                 break_or_not=True
				
				###########saving croppeds-samples Intensity VV############
				file_saved_sample = os.path.join(path_saved_sample,fl_name_saved+'.png')
				fig,ax = plt.subplots()
				ax.imshow(sample,cmap='gray',vmin=vmin,vmax=vmax)
				fig.savefig(file_saved_sample)
				fig.clear()
				plt.close()
				###########saving croppeds-samples Intensity VV############
	###########saving full-sample Intensity VV############
	file_saved_full_sample = os.path.join(path_saved__full_sample,day_sar_filename+'_full.png')
	fig,ax = plt.subplots(figsize=(20,10))
	ax.imshow(img_full_sample,cmap='gray',vmin=vmin,vmax=vmax)
	fig.savefig(file_saved_full_sample,dpi=300)
	fig.clear()
	plt.close()


