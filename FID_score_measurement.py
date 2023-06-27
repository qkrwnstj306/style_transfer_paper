import os, subprocess
from os import path
import shutil

def FID_measurement(model,style):
    origin_pth_dir = f'./output_img/{style}'
    trg_pth_dir = f'./output_img/{model}/{style}'
    copy_pth = f'./output_img/{model}/{style}/all_images'
    origin_copy_pth = f'./output_img/{style}/all_images'
    all_index_lst = os.listdir(trg_pth_dir)
    index_lst = [file for file in all_index_lst if not file.endswith(".npy")]
    
    for index in index_lst:
        file_pth_dir = trg_pth_dir + '/' + index
        #각 prompt index 별로 계산하고 싶을 때, 해당 코드 아래는 주석처리하면됨. 대신 txt 쓰는 거에서 index 추가 해야 됨
        # proc = subprocess.Popen([f"python ./PyTorch-FID-score/fid_score.py {origin_pth_dir} {file_pth_dir} --model {model} --style {style}"],  shell=True)
        # out, _ = proc.communicate() 
        # =============================
        file_lst = os.listdir(file_pth_dir)

        if not path.isdir(copy_pth):
            os.makedirs(copy_pth)
        
        for file in file_lst:
            if not path.exists(copy_pth + '/' + file):
                shutil.copy(file_pth_dir+'/'+file, copy_pth+'/'+file)
    
    all_index_lst = os.listdir(origin_pth_dir)        
    index_lst = [file for file in all_index_lst if not file.endswith(".npy")]
    for index in index_lst:
        file_pth_dir = origin_pth_dir + '/' + index
        file_lst = os.listdir(file_pth_dir)

        if not path.isdir(origin_copy_pth):
            os.makedirs(origin_copy_pth)
        
        for file in file_lst:
            if not path.exists(origin_copy_pth + '/' + file):
                shutil.copy(file_pth_dir+'/'+file, origin_copy_pth+'/'+file)
                
    proc = subprocess.Popen([f"python ./PyTorch-FID-score/fid_score.py {origin_copy_pth} {copy_pth} -c 0 --model {model} --style {style}"],  shell=True)
    _ , _ = proc.communicate()
    copy_file_lst = os.listdir(copy_pth)
    with open('fid_score.txt','a') as f:
        f.write(f'\n# of image : {len(copy_file_lst)}')
    for file in copy_file_lst:
        if path.exists(copy_pth+'/'+file):
            os.unlink(copy_pth+'/'+file)

    copy_file_lst2 = os.listdir(origin_copy_pth)
    for file in copy_file_lst2:
        if path.exists(origin_copy_pth+'/'+file):
            os.unlink(origin_copy_pth+'/'+file)
    
if __name__=="__main__":
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model_dic = {'0' : 're_img_0_500',
                '1' : 're_img_0_750',
                '2' : 're_img_0_1000', # training step 비교군 1
                '3' : 're_img_0_2000',
                '4' : 're_img_0_3000',
                
                '5' : 'in_img_face_with_background_750',
                '6' : 'in_img_face_with_background_1000',
                '7' : 'in_img_face_with_background_2000',
                '8' : 'in_img_face_with_background_3000', 
                
            }
    instance_style = { '1' : 'mid-journey', 
                      '2' : 'anime',
                      '3' : 'realistic'}
    Comparative_group_one = True
    Comparative_group_two = False
    if Comparative_group_one and Comparative_group_two == False:
        for i in range(1,3+1):
            style = instance_style[str(i)]
            for j in range(0,0+1): #range(1,3+1):
                model = model_dic[str(j)]    
                FID_measurement(model, style)
    
    elif Comparative_group_one==False and Comparative_group_two == True :
        for i in range(1, 3+1):
            style = instance_style[str(i)]
            
            model = model_dic[str(7)]
            FID_measurement(model, style)
        