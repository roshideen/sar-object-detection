import datetime
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
import img_utils
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

def get_sar_product_subset(s1_read,lat:float=9.126140086498603,long:float=99.20271266495655,width:int=5000,height:int=3000,do_tc:bool=True):
	x,y = img_utils.convert_latlong2pixel_sentinel1(s1_read,lat,long)
	x,y = int(x),int(y)
	subset =img_utils.get_subset_product(s1_read,x,y,width,height)
	band_nm =img_utils.get_band_names(subset)
	if do_tc:
		subset =img_utils.do_terrain_correction(subset,band_nm, downsample=False)
	del s1_read
	return subset,band_nm

def get_all_masked_band_GRDSAR(subset,gulf_mask):
	band1 = img_utils.project2img(subset,'Amplitude_VH') #int16
	band2 = img_utils.project2img(subset,'Intensity_VH') #int32
	band3 = img_utils.project2img(subset,'Amplitude_VV') #int16
	band4 = img_utils.project2img(subset,'Intensity_VV') #int32
	numerator = (band2*band4)
	denominator = (band1*band4)+(band2*band3)
	band5 = np.where(denominator==0,0,numerator/denominator)
	band_all_data = [band1,band2,band3,band4,band5]
	xlim = np.min([band4.shape[1],gulf_mask.shape[1]])
	ylim = np.min([band4.shape[0],gulf_mask.shape[0]])
	band_all_data = [data_i[:ylim,:xlim]*gulf_mask[:ylim,:xlim] for data_i in band_all_data]
	del band1, band2, band3, band4, band5,numerator, denominator
	return band_all_data

def get_split_bands(band_all_data,ksize):
	return [reshape_split(data_i,ksize) for data_i in band_all_data]

def tiling_saving_images(subset,split_band_all_data,full_img,df_kanumx,gulf_maskx,day_sar_filename,saved_paths,ksize,fontScale:int=3,thickness:int=10,color:int=5e3):
	df_kanum=df_kanumx.copy()
	gulf_mask = gulf_maskx.copy()
	tmp = full_img.copy()
	band_all_data = split_band_all_data.copy()
	path_saved_img, path_saved_label,path_saved_label,path_saved_sample, path_saved__full_sample = saved_paths
	



	column_kanum_tobekept = [i for i in df_kanum.columns if 'kanumFoundDate' in i]
	df_kanum.rename(columns={df_kanum.columns[0]:'kanum_id'},inplace=True)
	df_kanum.set_index('kanum_id',inplace=True)

	fmt = '%Y%m%d'
	date1 = datetime.strptime(day_sar_filename,fmt)
	datecolumn_fmt = [datetime.strptime(x.split('_')[1],fmt) for x in column_kanum_tobekept]
	gtdate_column = column_kanum_tobekept[(np.where((np.array([(date1-x).days for x in datecolumn_fmt])<=0))[0][0])]
	found_kanums = df_kanum.index[np.where(df_kanum.loc[:,gtdate_column].notnull())[0]]
	found_kanums_pixel = [df_kanum.loc[list(found_kanums),['latitude','longitude']].apply(lambda x: img_utils.convert_latlong2pixel_sentinel1(subset,*x),axis=1)]
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
	            ###########saving bands############
	            file_saved_cropped_bands = os.path.join(path_saved_img,fl_name_saved+'.npy')
	            saved_bands_data_npy =  h5py.File(file_saved_cropped_bands, "w")
	            saved_bands_data_npy.create_dataset("sar", np.shape(cropped_all_bands), dtype=np.float64, data=cropped_all_bands)
	            ###########saving bands############
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


