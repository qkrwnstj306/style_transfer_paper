import os

def make_dic(model_dic, model_style, prompt_id):
    file_path = f'./output_img/{model_dic}/{model_style}/{prompt_id}'
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    return file_path
    