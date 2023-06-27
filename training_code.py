import os
import subprocess 
import json
import torch 
import requests
#드림부스 다운
'''
subprocess.call('wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py',shell=True)

subprocess.call('wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py',shell=True)

subprocess.call('pip install -qq git+https://github.com/ShivamShrirao/diffusers',shell=True)

subprocess.call('pip install -q -U --pre triton',shell=True)

subprocess.call('pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers',shell=True)

#####허깅페이스 경로
path='/home/kkko/capston_design/.huggingface'
os.mkdir(path)
'''

def training_dreambooth(gpu, instance_prompt, model_dic, instance_style, class_img_dic):
    #GPU setting
    #torch.cuda.set_device(int(gpu))
    
    HUGGINGFACE_TOKEN = "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"
    os.system(f'echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token')

    #base model
    #맨 처음 model weight를 가져오는 경로
    #MODEL_NAME = f"./civit_model_diffusers/{model}" #@param {type:"string"}
    #이후 model weight 가져 오는 경로
    MODEL_NAME = f"./model_weight/SD"
    
    #weight가 저장될 경로
    OUTPUT_DIR = f"./model_weight/{model_dic}/{instance_style}"
    print(f"[*] Weights will be saved at {OUTPUT_DIR}")
    
    # You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.
    concepts_list = [
        {
            "instance_prompt":      f"a photo of {instance_prompt} style",
            "class_prompt":         "a photo of style",
            
            "instance_data_dir":    f"./instance_img/{instance_style}",
            "class_data_dir":       f"./class_img/{class_img_dic}"
        }
    ]

    # `class_data_dir` contains regularization images

    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)
        
    #vae : stabilityai/sd-vae-ft-mse ,./anime_vae/anything
    subprocess.call(f'python3 train_dreambooth.py \
    --pretrained_model_name_or_path={MODEL_NAME} \
    --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
    --output_dir={OUTPUT_DIR} \
    --revision="fp16" \
    --with_prior_preservation --prior_loss_weight=0.0 \
    --seed=1337 \
    --resolution=512 \
    --train_batch_size=1 \
    --train_text_encoder \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=1 \
    --sample_batch_size=0 \
    --max_train_steps=500 \
    --save_interval=10000 \
    --concepts_list="./concepts_list.json"',shell=True)

    print("instance prompt : ", instance_prompt)
    

if __name__=='__main__':
    instance_prompt = 'zwx'
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
    class_img_dic = {'1' : 'mid-journey',
                     '2' : 'anime',
                     '3' : 'realistic',
                     '4' : 'test',#65
                     '5' : 'test2'}  # 146
    
    training_dreambooth(3, instance_prompt, model_dic['0'], instance_style['1'], class_img_dic['1'])