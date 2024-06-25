import pandas as pd
from transformers import AutoModelForSeq2SeqLM # import LLM
from transformers import AutoTokenizer  # tokenizer
import torch
gpu_num=4
# from transformers import GenerationConfig   # configurator
use_cuda=True
if use_cuda:
    cuda_device = torch.device(f'cuda:{gpu_num}')
    print(cuda_device)
    print('check 1')
    # print(torch.cuda.mem_get_info())

else:
    cuda_device = torch.device('cpu')
    print('cpu')
path='./data/'
model_name='./model/flan-t5-base/'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(cuda_device)
# He loves the games we have on the IPAD and it helps him with his letters and numbers too.
# We go to the library to take out cd's and also audio books with pictures.
text="He loves the games we have on the IPAD and it helps him with his letters and numbers too."


# prompt = f"""
# The following strategies belong to the same category as environment/context.
# Example 1: We go to restaurants I know our son will sit down and eat.
# Example 2: He sees his brothers doing the same basic care routines so he usually likes to join in.
# Generate more example strategies of environment/context category:
#     """
# print(prompt)
# Input constructed prompt instead of the dialogue.

# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.1)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)
def save_to_file(text):
    modelname=model_name.split("/")[2]
    with open(path+modelname+'_augment.txt', mode='a', encoding='utf-8') as myfile:
        for lines in text:
            myfile.write(lines )
            myfile.write('\n')
    print('done')
def generate_augment(text,category,context,setting,content,runs):
    prompt= f"""
        Here is an example of {category} strategy:
        {text}

        Please generate rewrite of the above strategy  keeping the style similar. 
        """
    prompt_context=f"""
    Here is an example of {category} strategy in context of {context}:
    {text}
    
    Please generate rewrite of the above strategy  keeping the style similar. 
    """

    prompt_context_setting = f"""
        Here is an example of {category} strategy in context of {context} in {setting} setting:
        {text}

        Please generate rewrite of the above strategy  keeping the style similar. 
        """
    augmented_text=[]
    for t in [0.8,0.9,1.0]:
        # generation_config = GenerationConfig(max_new_tokens=512, do_sample=True, temperature=t)  #LOL
        for run in range(runs):

            inputs = tokenizer(prompt, return_tensors='pt')
            output = tokenizer.decode(
                model.generate(
                    inputs["input_ids"].to(cuda_device),
                    max_length=128, do_sample=True,temperature=t
                )[0],
                skip_special_tokens=True
            )

            if output!=text:
                augmented_text.append(content+'\t'+output)
                print('prompt')
                print(output)

            inputs = tokenizer(prompt_context, return_tensors='pt')
            output = tokenizer.decode(
                model.generate(
                    inputs["input_ids"].to(cuda_device),
                    max_length=128, do_sample=True, temperature=t
                )[0],
                skip_special_tokens=True
            )

            if output != text:
                augmented_text.append(content + '\t' + output)
                print('prompt_context')
                print(output)

            inputs = tokenizer(prompt_context_setting, return_tensors='pt')
            output = tokenizer.decode(
                model.generate(
                    inputs["input_ids"].to(cuda_device),
                    max_length=128, do_sample=True, temperature=t
                )[0],
                skip_special_tokens=True
            )

            if output != text:
                augmented_text.append(content + '\t' + output)
                print('prompt_context_setting')
                print(output)
    # save_to_file(augmented_text)
train=pd.read_csv(path+'train.csv')
print(train.columns.tolist())
category=["environment/context", "Sense of Self", "Preferences", "Activity Competence", "non-strategy"]
for idx, row in train.iterrows():
    content=row['Dataset']+'\t'+str(row['ID'])+'\t'+row['broad activity ']+'\t'+row[ 'setting ']+'\t'+str(row['label'])+'\t'+row['strategy']
    # if row['label'] in [1,2,3]:
    #     runs=3
    # else:
    #     runs=2
    generate_augment(row['strategy'], category[int(row['label'])], category[int(row['broad activity '])],category[int(row['setting '])],content, 1)
    if idx==10:
        break
# for t in [0.8,0.9,1.0]:
#     outputs = model.generate(**inputs,max_length=150,no_repeat_ngram_size=5, do_sample=True,temperature=t)
#     print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#
