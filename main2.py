import SD
import cmd_txt2img
import argparse

def cmd_generation_img(model_dic, model_style):
    for model in model_dic:
                #style만큼 실행
                for style in model_style:
                        # prompt 실험 개수만큼 실행
                        index = 1    
                        for prompt_index in prompt_id:
                            prompt = prompt_id[prompt_index]
                            prompt =  'a photo of zwx style, '+prompt
                            cmd_txt2img.main_process(prompt, negative_prompt, model, style, int(prompt_index), index, gpu)
#output_img -> model_name -> prompt_id path에 저장
if __name__=="__main__":
    #instance에 사람 얼굴만 들어갔을 때와 아닐 때의 비교 실험
    INSTANCE_ONLY_FACE_VERSE_INCLUDING_BACKGROUND = False
    #training step에 따른 실험, no re image
    TRAINING_STEP_COMPARISON = True
    #regularization image가 들어갔을 때, training step을 끌어올릴수 있다는 걸 검증하는 실험
    ADD_REGULARIZATION = False

    prompt_id = dict()
    with open('./prompt.txt','r') as f:
        prompt_lst = f.readlines()
    
        for i in range(len(prompt_lst)):
            
            prompt_id[str(i+1)] = prompt_lst[i].rstrip('\n')
        
    negative_prompt = '(worst quality, low quality:1.2), canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), 3d render'
    gpu = 3
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type= int, default= 1)  
    args = parser.parse_args()
    
    for i in range(args.iter):
        if INSTANCE_ONLY_FACE_VERSE_INCLUDING_BACKGROUND== True and TRAINING_STEP_COMPARISON == False and ADD_REGULARIZATION==False:
            model_dic = ['in_img_only_face', 'in_img_only_background', 'in_img_face_with_background']
            model_style = ['anime', 'mid-journey', 'realistic']
            
            cmd_generation_img(model_dic, model_style)
                            
        elif INSTANCE_ONLY_FACE_VERSE_INCLUDING_BACKGROUND== False and TRAINING_STEP_COMPARISON == True and ADD_REGULARIZATION==False:
            model_dic = ['re_img_0_500'] #'re_img_0_500','re_img_0_750','re_img_0_1000', 're_img_0_2000', 're_img_0_3000', 're_img_0_4000',  
            model_style = ['anime' ,'realistic' ,'mid-journey'] 
            
            cmd_generation_img(model_dic, model_style)
            
        elif ADD_REGULARIZATION and INSTANCE_ONLY_FACE_VERSE_INCLUDING_BACKGROUND == False and TRAINING_STEP_COMPARISON == False:
            model_dic = ['in_img_face_with_background_750'] # 'in_img_face_with_background_3000'
            model_style = ['anime','mid-journey', 'realistic']  #'anime', 'mid-journey', 'realistic'
            
            cmd_generation_img(model_dic, model_style)
                            
        elif INSTANCE_ONLY_FACE_VERSE_INCLUDING_BACKGROUND== False and TRAINING_STEP_COMPARISON == False and ADD_REGULARIZATION==False:
            model_dic = ['SD','re_img_50_face', 're_img_0']
            model_style = ['mid-journey','anime', 'realistic']
            
            for model in model_dic:
                #style만큼 실행
                if model=='SD':
                    for style in model_style:
                        index = 1    
                        # prompt 실험 개수만큼 실행
                        for prompt_index in prompt_id.keys():
                            prompt = prompt_id[prompt_index]
                            prompt = style + ' style, ' + prompt
                            SD.main_process(prompt, negative_prompt, model, style, int(prompt_index), index, gpu)
                            index = index + 1
                
                else :
                    for style in model_style:
                        # prompt 실험 개수만큼 실행
                        index = 1    
                        for prompt_index in prompt_id:
                            prompt = prompt_id[prompt_index]
                            prompt = 'in the stlye of zwx, ' + prompt 
                            cmd_txt2img.main_process(prompt, negative_prompt, model, style, int(prompt_index), index, gpu)
                            