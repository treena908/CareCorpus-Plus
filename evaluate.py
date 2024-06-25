#import
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
model_name ="bert-base-uncased"
use_cuda=False #whether use gpu
gpu_num=0

#model path, I supplied the best_model.pth file, u can add the file path as per your file location

checkpoint_dir ='./output/bert-base-uncased_16_10_multi_extended_augmented_flant5-xl_pvi_perintent_True/checkpoint/seed1/'
if use_cuda:
    device = torch.device(f'cuda:{gpu_num}')
    print(device)
else:
    device = torch.device('cpu')
    print('cpu')
def prepare_dataloader(text):

  test_input_ids, test_att_masks = encode([text])

  test_input_ids = torch.LongTensor(test_input_ids).to(device)
  test_att_masks = torch.LongTensor(test_att_masks).to(device)

  return  test_input_ids, test_att_masks

def load_model_from_ckpoint(model):

    try:
        if os.path.isfile(os.path.join(checkpoint_dir, "best_model.pth")):
          print('loading saved model from %s'%(checkpoint_dir))

          # create new OrderedDict that does not contain `module.`
          checkpoint = torch.load(os.path.join(checkpoint_dir, "best_model.pth"), map_location=torch.device("cpu"))


          from collections import OrderedDict

          new_state_dict = OrderedDict()
          for key, v in checkpoint.items():
              name = key.replace("module.", "")  # remove `module.`
              new_state_dict[name] = v

          # load params
          model.load_state_dict(new_state_dict, strict=False)
          return model
    except Exception as e:
        print(e)
      # print(model)
def encode(docs):
  '''
  This function takes list of texts and returns input_ids and attention_mask of texts
  '''
  encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=300, padding='max_length',
                                             return_attention_mask=True, truncation=True, return_tensors='pt')
  input_ids = encoded_dict['input_ids']
  attention_masks = encoded_dict['attention_mask']
  return input_ids, attention_masks


def initialize_model(class_labels):
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = BertForSequenceClassification.from_pretrained(model_name,
                                                        num_labels=class_labels,
                                                        output_attentions=False,
                                                        output_hidden_states=False)
  return model,tokenizer
def prepare_dataloader(text):
  # encode input text to be appropriate for bert model
  test_input_ids, test_att_masks = encode([text])
  test_dataset = TensorDataset(test_input_ids, test_att_masks)
  test_sampler = SequentialSampler(test_dataset)
  test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)
  return test_dataloader
def predict(text):
  '''
  input: text (string)
  output: predicted class (integer)

  '''


  #prepare data for the model
  test_dataloader=prepare_dataloader(text)
  # test_input_ids, test_att_masks = encode(text,tokenizer)
  # test_dataset = TensorDataset(test_input_ids, test_att_masks)
  # test_dataloader = DataLoader(test_dataset, batch_size=1)
  with torch.no_grad():
      for step_num, batch_data in enumerate(test_dataloader):
        input_ids, att_mask = [data.to(device) for data in batch_data]
        output = model(input_ids=input_ids, attention_mask=att_mask)


        logits=output.logits.cpu().detach().numpy()
        output_class=np.argmax(logits, axis=-1) #predicted class of the text




  return output_class[0]
#here are some sameple text

texts=[


    "Not having participation rules so my son can not join in.",
    "making the work a game",
 "Using tablet at school- teachers use hand over hand to show him how to do the activity.",

    "Allow chance for child to choose events therefore encourage participation.",
"Yoga- help him stretch/excercise",

"allow independence.",
"playing.",
"allow her to be creative",
"123.",
    " "


]
#driver code
#initialise model
model, tokenizer = initialize_model(class_labels=5)
model.to(device)
model.eval()
#load saved model
model=load_model_from_ckpoint(model)

for t in texts:
    #calling the prediction function
    result=predict(text=t)
    #print class label
    print(result)
