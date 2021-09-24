import argparse
from datetime import datetime
from snappy import ProductIO,HashMap,GPF
import os,gc
import snappy
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson
import cv2
import glob
import h5py
import numpy as np
import img_utils
import img_cropped_sar_preparing
from utils.general import non_max_suppression
import matplotlib.pyplot as plt
import torch
import pandas as pd
import pickle

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--product_path',type=str)
    # parser.add_argument('--index_sar', type=int, help='index of sar-file')
    parser.add_argument('--sar_filename',type=str)
    parser.add_argument('--ksize',type=int, help='image size')
    parser.add_argument('--weight_no',type=str, help='')
    parser.add_argument('--ninchannel', type=int, help='no. of input channel (1,4)')
    parser.add_argument('--whatband', type=int, default=4, help='band selected for 1 channel[]0,4], currently support only (3,4)')
    parser.add_argument('--usebestmodel',type=bool,default=True)
    parser.add_argument('--conf_thres',type=float,default=0.25)
    parser.add_argument('--iou_thres',type=float,default=0.45)
    parser.add_argument('--max_det',type=float,default=1000)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def get_pixel_kanum(dfx,found_kanums,proj):
    df_kanum = dfx.copy()
    found_kanums_pixel = [df_kanum.loc[list(found_kanums),['latitude','longitude']].apply(lambda x: img_utils.convert_latlong2pixel_sentinel1(proj,*x),axis=1)]
    found_kanums_pixel = list(found_kanums_pixel[0].values)
    del df_kanum, proj, dfx
    return found_kanums_pixel

def get_df_pixel(ans,flag_fromwhat:int=1):
    if flag_fromwhat == 1: #for gt kanum
        df_gt = pd.DataFrame(columns=['pixel_x','pixel_y'])
        count_idx = 0
        for i in ans['kanum_gt_pixel']:
            for j in i:
                df_gt.loc[count_idx,'pixel_x']=j[0]
                df_gt.loc[count_idx,'pixel_y']=j[1]
                count_idx+=1
        return df_gt
    if flag_fromwhat == 2: #for predicted kanum
        df_model = pd.DataFrame(columns=['pixel_xy'])
        count_idx = 0
        for i in ans['model_xywh']:
            df_model.loc[count_idx,'pixel_xy']=(i[0],i[1])
            count_idx+=1
        return df_model
    
def find_kanum_distance_gt_pred(p1,dfx):
    p_all = dfx[['pixel_x','pixel_y']].values
    min_dist = np.sqrt(np.sum(np.abs(p1-p_all)**2,axis=1).astype(int)).min()
    argmin_dist = np.sqrt(np.sum(np.abs(p1-p_all)**2,axis=1).astype(int)).argmin()
    return argmin_dist,min_dist

def checking_predicted_gt_kanum(df_model,df_gt):
    df_model['result'] = df_model['pixel_xy'].apply(lambda x:find_kanum_distance_gt_pred(x,df_gt))
    df_model['argmin_dist_gt'] = df_model['result'].apply(lambda x:x[0])
    df_model['min_dist'] = df_model['result'].apply(lambda x:x[1])
    df_model['status_plot'] = 0
    df_model.loc[df_model[df_model['min_dist']<=3].index,'status_plot']=1
    df_gt['status_plot']=0
    df_gt.loc[df_model[df_model['status_plot']==1]['argmin_dist_gt'].unique(),'status_plot']=1
    return df_model,df_gt

def get_tiling(bandx,mask_limits,ninchannel:int=1,ksize:int=256):
    tiled_imgs = []
    tiled_pixels = []
    xmsk_white_min, xmsk_white_max,ymsk_white_min, ymsk_white_max = mask_limits
    for i in range(0,bandx.shape[0]):
        for j in range(0,bandx.shape[1]):
            y1,y2 = i*ksize,(i+1)*ksize
            x1,x2 = j*ksize,(j+1)*ksize
            if ((x2>xmsk_white_min) and (y2>ymsk_white_min)) or ((xmsk_white_min<x1<xmsk_white_max) and (y2>ymsk_white_min)):
                if ninchannel == 1:
                    tiled_img = bandx[i,j,:,:].copy()
                    tiled_img = tiled_img / 21.37616498108423
                    tiled_img =  np.expand_dims(tiled_img,0)
                elif ninchannel == 4:
                    tiled_img = np.zeros((ninchannel,ksize,ksize))
                    tiled_img[0,:,:] = bandx[0][i,j].copy() / 30.3733705341102 #Amplitude VH
                    tiled_img[1,:,:] = bandx[1][i,j].copy() / 974.8808288085606 #Intensity VH
                    tiled_img[2,:,:] = bandx[2][i,j].copy() / 73.68063162696058 #Amplitude VV
                    tiled_img[3,:,:] = bandx[3][i,j].copy() / 5825.10278339073 #Intensity VV        
                tiled_imgs.append(tiled_img)
                tiled_pixels.append((x1,y1))
    return tiled_imgs,tiled_pixels
    
def get_corrected_pixel(i,det,tiled_pixels,xyxys,xywhs):
    xyxy = det[:,:5].cpu().numpy()
    xywh = xyxy2xywh(xyxy)

    xyxy_corrected_pixel = xyxy.copy()
    xyxy_corrected_pixel[:,0] = xyxy_corrected_pixel[:,0]+tiled_pixels[i][0]
    xyxy_corrected_pixel[:,1] = xyxy_corrected_pixel[:,1]+tiled_pixels[i][1]
    xyxy_corrected_pixel[:,2] = xyxy_corrected_pixel[:,2]+tiled_pixels[i][0]
    xyxy_corrected_pixel[:,3] = xyxy_corrected_pixel[:,3]+tiled_pixels[i][1]

    xywh_corrected_pixel = xywh.copy()
    xywh_corrected_pixel[:,0] = xywh_corrected_pixel[:,0]+tiled_pixels[i][0]
    xywh_corrected_pixel[:,1] = xywh_corrected_pixel[:,1]+tiled_pixels[i][1]
    xyxys = np.append(xyxys,xyxy_corrected_pixel,axis=0)
    xywhs = np.append(xywhs,xywh_corrected_pixel,axis=0)
    return xyxys, xywhs

def get_sen_prec(df_gt,df_model):
    n_actual_pos = df_gt.shape[0]
    n_pred_pos = df_model.shape[0]
    tp = df_model[df_model['status_plot']==1].shape[0]
    fp = n_pred_pos-tp
    fn = n_actual_pos-tp
    sen = tp/(tp+fn)
    prec = tp/(tp+fp)
    return sen, prec
        
        

def main(opt):
    # print(opt.sar)
    ## Load gulf mask

    ksize = opt.ksize
    # product_path = opt.product_path
    # index_sar = opt.index_sar
    sar_filename = opt.sar_filename
    weights_no = opt.weight_no
    ninchannel = opt.ninchannel
    usebestmodel = opt.usebestmodel
    whatband = opt.whatband
    print('opt: ',opt)

    mask_path = r'D:\08 aimagin\satellite image project\sar'
    gulf_mask = cv2.imread(os.path.join(mask_path,'gulf_mask_SAR2.png'),-1)
    gulf_mask[gulf_mask==255]=1
    ## Load SAR
    # product_path = r'E:\aimagin\sar_images\sentinel-1_projects'
    # input_s1_files = sorted(list(glob.glob(os.path.join(product_path,'*S1*.zip'),recursive=True)))
    # sar_filename = input_s1_files[index_sar]
    # print(sar_filename)
    day_sar_filename = sar_filename.split('\\')[-1].split('_')[4].split('T')[0]
    s1_read = snappy.ProductIO.readProduct(sar_filename)
    subset,band_nm = img_cropped_sar_preparing.get_sar_product_subset(s1_read)
    band_all_data = img_cropped_sar_preparing.get_all_masked_band_GRDSAR(subset,gulf_mask)
    band_all_data = img_cropped_sar_preparing.get_split_bands(band_all_data,ksize)
    if ninchannel == 1:
        bandx = band_all_data[whatband]
        range_final_i = bandx.shape[0]
        range_final_j = bandx.shape[1]
    elif ninchannel == 4:
        bandx = band_all_data[:4]
        range_final_i = bandx[0].shape[0]
        range_final_j = bandx[0].shape[1]
    xmsk_white_min, xmsk_white_max = np.where(gulf_mask==1)[1].min(),np.where(gulf_mask==1)[1].max()
    ymsk_white_min, ymsk_white_max = np.where(gulf_mask==1)[0].min(),np.where(gulf_mask==1)[0].max()
    
    mask_limits = xmsk_white_min, xmsk_white_max,ymsk_white_min, ymsk_white_max
    tiled_imgs,tiled_pixels = get_tiling(bandx,mask_limits,ninchannel=ninchannel,ksize=ksize)

    imgs_batch = np.zeros((len(tiled_pixels),1,ksize,ksize))
    for i,j in enumerate(tiled_imgs):
        imgs_batch[i,:] = j
        
    imgs_batch_tensor = torch.from_numpy(imgs_batch)
    

    ###########################
    ###kanum groundtruth#######
    kanum_path = r'D:\08 aimagin\satellite image project\sat\coordinate_khanum'
    kanum_file = 'kanum_map'+'.csv'
    df_kanum = pd.read_csv(os.path.join(kanum_path,kanum_file))
    column_kanum_tobekept = [i for i in df_kanum.columns if 'kanumFoundDate' in i]
    df_kanum.rename(columns={df_kanum.columns[0]:'kanum_id'},inplace=True)
    df_kanum.set_index('kanum_id',inplace=True)

    fmt = '%Y%m%d'
    date1 = datetime.strptime(day_sar_filename,fmt)
    datecolumn_fmt = [datetime.strptime(x.split('_')[1],fmt) for x in column_kanum_tobekept]
    gtdate_column = column_kanum_tobekept[(np.where((np.array([(date1-x).days for x in datecolumn_fmt])<=0))[0][0])]
    print('\tdate of SAR project: ',day_sar_filename)
    print('\tdate found on kanum df: ',gtdate_column)

    found_kanums = df_kanum.index[np.where(df_kanum.loc[:,gtdate_column].notnull())[0]]
    found_kanums_1 = df_kanum.index[np.where(df_kanum.loc[:,gtdate_column]==1)[0]]
    found_kanums_2 = df_kanum.index[np.where(df_kanum.loc[:,gtdate_column]==2)[0]]
    found_kanums_pixel_1 = get_pixel_kanum(df_kanum,found_kanums_1,subset)
    found_kanums_pixel_2 = get_pixel_kanum(df_kanum,found_kanums_2,subset)

 #    ## Loading Model
    path_weight = r'D:\08 aimagin\satellite image project\python\yolo\yolov5\runs\train'

    weights_no = weights_no.split(',')
    for weight_no in weights_no:
        if usebestmodel:
            weight = os.path.join(path_weight,weight_no,'weights','best.pt')
        else:
            weight = os.path.join(path_weight,weight_no,'weights','last.pt')
        path_yolo_model = r'D:\08 aimagin\satellite image project\python\yolo\yolov5'
        model = torch.hub.load(path_yolo_model,'custom',path=weight,source='local')

        conf_thres = opt.conf_thres
        iou_thres = opt.iou_thres
        max_det=opt.max_det
        classes = None
        agnostic_nms = None
        
        count_kanum=0
        model.eval()
        kanum_founds = []
        img_founds = []
        
        stats = {
                'sar_date':[],
                'gt_kanum_date':[],
                'conf_thres':[],
                'iou_thres':[],
                'sensitivity':[],
                'precision':[]}

        for conf_thres in [0.05,0.15,0.25,0.45,0.55]:#np.arange(0,1,0.02):
            for iou_thres  in [0.25,0.45,0.65]:#np.arange(0,1,0.02):
                xyxy_all = np.empty((0,5),float)
                xywh_all = np.empty((0,5),float)
                pred = model(imgs_batch_tensor)[0]
                pred_nms = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, agnostic=agnostic_nms, max_det=max_det)   
                for i,det in enumerate(pred_nms):
                    if det.shape[0]:
                        xyxy_all, xywh_all = get_corrected_pixel(i,det,tiled_pixels,xyxy_all,xywh_all)
                ans = {'model_xywh':xywh_all,'kanum_gt_pixel':[found_kanums_pixel_1,found_kanums_pixel_2]}#,null_kanums_pixel
                df_gt = get_df_pixel(ans,flag_fromwhat=1)
                df_model = get_df_pixel(ans,flag_fromwhat=2)
                df_model_checked,df_gt_checked = checking_predicted_gt_kanum(df_model,df_gt)
                sen, prec = get_sen_prec(df_gt,df_model)
                stats['sar_date'].append(day_sar_filename)
                stats['gt_kanum_date'].append(gtdate_column)
                stats['conf_thres'].append(conf_thres)
                stats['iou_thres'].append(iou_thres)
                stats['sensitivity'].append(sen)
                stats['precision'].append(prec)

        df_stats = pd.DataFrame.from_dict(stats)
        folder_big = 'plot_yolo_detection_result'
        df_stats.to_csv(os.path.join(folder_big,'tuning_conf_iou_thres','model_'+weight_no+'_'+day_sar_filename+'.csv'))
    # folder_saved = 'yolo_detected_model_'+weight_no+'_'+day_sar_filename+'_gtDate_'+gtdate_column+'_bestmodelornot_'+str(usebestmodel)
    # if folder_saved not in os.listdir(folder_big):
    #   os.mkdir(os.path.join(folder_big,folder_saved))


# saved_result = {'model_xywh':xywh_all,
#                     'kanum_gt_pixel':[found_kanums_pixel_1,found_kanums_pixel_2]}
#     ans = saved_result

        
    # test_img = np.zeros((range_final_i*ksize,range_final_j*ksize),dtype=np.float32)
    # for i,(j,k) in enumerate(tiled_pixels):
    #   if ninchannel == 1:
    #       test_img[k:k+ksize,j:j+ksize] = tiled_imgs[i]
    #   elif ninchannel == 4:
    #       test_img[k:k+ksize,j:j+ksize] = tiled_imgs[i][0]


    
    


     
    # # df_gt = get_df_pixel(ans,flag_fromwhat=1)
    # # df_model = get_df_pixel(ans,flag_fromwhat=2)
    # # df_model_checked,df_gt_checked = checking_predicted_gt_kanum(df_model,df_gt)

    # print('\t#kanum found: '+str(df_model.shape[0]))
    # print('\t#kanum found corrected: '+str(df_model[df_model['status_plot']==1].shape[0]))
    # print('\t#total labeled kanums: '+str(df_gt.shape[0]))
    # print('\t#type_1 kanums: '+str(len(found_kanums_1))+', #type_2 kanums: '+str(len(found_kanums_2)))

    # test_img_tmp = cv2.convertScaleAbs(test_img,alpha=30)
    # test_img_tmp = cv2.cvtColor(test_img_tmp,cv2.COLOR_GRAY2RGB)
    # for i in range(0,df_model.shape[0]):
    #   pixel_x,pixel_y,status_plot = int(df_model['pixel_xy'].iloc[i][0]),int(df_model['pixel_xy'].iloc[i][1]),df_model['status_plot'].iloc[i]
    #   if status_plot == 0:
    #       color = (255,0,0)
    #   if status_plot == 1:
    #       color=(0,255,0)
    #   cv2.circle(test_img_tmp,(pixel_x,pixel_y),radius=8,color=color,thickness=2)
    # for i in list(df_gt[df_gt['status_plot']==0].index):
    #   pixel_x,pixel_y = int(df_gt['pixel_x'].loc[i]),int(df_gt['pixel_y'].loc[i])
    #   color = (0,255,255)
    #   cv2.circle(test_img_tmp,(pixel_x,pixel_y),radius=8,color=color,thickness=2)

    
    # ###########################

    # folder_big = 'plot_yolo_detection_result'
    # folder_saved = 'yolo_detected_model_'+weight_no+'_'+day_sar_filename+'_gtDate_'+gtdate_column+'_bestmodelornot_'+str(usebestmodel)
    # if folder_saved not in os.listdir(folder_big):
    #   os.mkdir(os.path.join(folder_big,folder_saved))

    # path_saved = os.path.join(folder_big,folder_saved,'plot.png')
    # fig,ax = plt.subplots(figsize=(20,10))
    # plt.imshow(test_img_tmp)
    # ax.imshow(test_img_tmp)
    # fig.savefig(path_saved,dpi=300)
    # fig.clear()
    # plt.close()

    # with open(os.path.join(folder_big,folder_saved,'modelpredicted_results.pkl'),'wb') as f:
    #   pickle.dump(saved_result,f)
    # df_model.to_csv(os.path.join(folder_big,folder_saved,'modelDf.csv'))
    # df_gt.to_csv(os.path.join(folder_big,folder_saved,'gtDf.csv'))    

    



            
            





if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

