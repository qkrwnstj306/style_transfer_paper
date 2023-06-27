import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import gc, subprocess
import img_rename
from random import randrange

def setting(model_dic, gpu = 0):
    #GPU setting
    torch.cuda.set_device(int(gpu))
    
    vae = AutoencoderKL.from_pretrained(f'./model_weight/SD', subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(f'./model_weight/SD', subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(f'./model_weight/SD', subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(f'./model_weight/SD', subfolder="unet")
    scheduler = PNDMScheduler.from_pretrained(f'./model_weight/SD', subfolder="scheduler")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(f'./model_weight/SD', subfolder="feature_extractor")
    
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=f'./model_weight/SD',
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
def txt2img(model_dic, pipe, prompt, negative_prompt, model_style, prompt_id, index):
   #height, width
    gc.collect()
    torch.cuda.empty_cache()
    
    #seed
    seed = randrange(300000000)
    
    print('user prompt : ',prompt)
    print('prompt id : ',prompt_id)
   
    num_samples = 2
    guidance_scale = 8
    num_inference_steps = 100
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
    
    
    file_path = img_rename.make_dic(model_dic, model_style, prompt_id)
    
    for img in images:
        img.save(file_path + f'/{index}.{seed}.png','PNG')
        index = index + 1
def main_process(prompt, negative_prompt, model_dic, model_style,prompt_id, index, gpu):
    
    pipe = setting(model_dic, gpu)
    txt2img(model_dic,pipe, prompt, negative_prompt, model_style, prompt_id, index)
    