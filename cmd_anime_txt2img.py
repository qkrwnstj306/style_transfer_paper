import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import gc, subprocess
import img_rename2, os
from random import randrange
import argparse

def setting(model_dic,  gpu = 3):
    #GPU setting
    #torch.cuda.set_device(int(gpu))
    vae = AutoencoderKL.from_pretrained(f'../capston_design/anime_vae', subfolder="vae",local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(f'./model_weight/{model_dic}', subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(f'./model_weight//{model_dic}', subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(f'./model_weight//{model_dic}', subfolder="unet")
    scheduler = PNDMScheduler.from_pretrained(f'./model_weight/{model_dic}', subfolder="scheduler")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(f'./model_weight/{model_dic}', subfolder="feature_extractor")
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=f'./model_weight/{model_dic}',
                                               vae=vae,
                                               text_encoder=text_encoder,
                                               tokenizer=tokenizer,
                                               unet=unet,
                                               scheduler=scheduler,
                                               safety_checker=None,
                                               feature_extractor=feature_extractor,
                                               torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    
    return pipe
def txt2img(model_dic, pipe, prompt, negative_prompt, prompt_id, index):
   #height, width
    gc.collect()
    torch.cuda.empty_cache()
    
    #seed
    seed = randrange(300000000)
    
    print('user prompt : ',prompt)
    print('prompt id : ',prompt_id)
   
    num_samples = 2
    guidance_scale = 8
    num_inference_steps = 50
    height = 512 
    width = 512 

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed)
        ).images
    
    
    file_path = img_rename2.make_dic(model_dic, prompt_id)
    
    for img in images:
        img.save(file_path + f'/{index}.{seed}.png','PNG')
        index = index + 1
def main_process(prompt, negative_prompt, model_dic, prompt_id, index, gpu):
    
    pipe = setting(model_dic, gpu)
    txt2img(model_dic,pipe, prompt, negative_prompt, prompt_id, index)
    
if __name__=='__main__':
    os.system('mkdir -p ~/.huggingface')
    HUGGINGFACE_TOKEN = "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"
    os.system(f'echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token')
    
    
    prompt_id = dict()
    with open('./prompt.txt','r') as f:
        prompt_lst = f.readlines()
    
        for i in range(len(prompt_lst)):
            
            prompt_id[str(i+1)] = prompt_lst[i].rstrip('\n')
    negative_prompt = '((hands)), (worst quality, low quality:1.4), canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render'
    model_dic = 'pastelboys'

    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type= int, default= 1)  
    args = parser.parse_args()
    
    for i in range(args.iter):
        index = 0 
        gpu = 2
        for prmpt_id in prompt_id.keys():
            main_process(prompt_id[prmpt_id], negative_prompt, model_dic, prmpt_id, index, gpu)