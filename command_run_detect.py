import os
import glob

path1 = r'E:\aimagin\sar_images\sentinel-1_projects'
path2 = r'E:\aimagin\sar_images\sentinel-1_projects2'
weight = 'exp207,'

for product_path in [path1,path2]:
    input_s1_files = sorted(list(glob.glob(os.path.join(product_path,'*S1*.zip'),recursive=True)))
    for i in range(0,len(input_s1_files)):
        sar_filename = input_s1_files[i]
        os.system('python 07_detect_kanum.py --sar_filename {} --ksize 256 --weight_no {} --ninchannel 1 --whatband 4 '.format(sar_filename,weight))