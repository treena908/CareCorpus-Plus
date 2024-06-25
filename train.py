import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer

import os
import sys
from sklearn.metrics.cluster import contingency_matrix
from collections import Counter
import  torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
# from transformers import Trainer, TrainingArguments
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, classification_report
from torch.nn.utils import clip_grad_norm_
import math
#############initialize params################
model_name ="bert-base-uncased"
# model_name = "emilyalsentzer/Bio_ClinicalBERT"

max_length = 300
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 2e-6
use_cuda=True
gpu_num=0
weighted_loss=True
do_train=False
do_test=True
load_model=True
col_name='strategy' #which column to select
# col_name='augmented_strategy' #which column to select for null
if use_cuda:
    device = torch.device(f'cuda:{gpu_num}')
    print(device)
else:
    device = torch.device('cpu')
    print('cpu')

# mode='binary'
# mode='ex_ins'
mode='multi'
# dataset='extended_forum_'
# dataset='extended_null_strategy'
dataset='extended_augmented_flant5-xl_pvi_perintent'
# dataset='forum_undersample'
path='./data/'
checkpoint_dir = './output/'+model_name+'_'+str(16)+'_'+str(EPOCHS)+'_'+mode+'_'+dataset+'_'+str(weighted_loss)+'/checkpoint/'

# checkpoint_dir = './output/'+model_name+'_'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+mode+'_'+dataset+'_'+str(weighted_loss)+'/checkpoint/'
# checkpoint_dir = './output/'+model_name+'_'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+mode+'_'+dataset+'/checkpoint/'
def save_to_file(text):
    if '/' in model_name:
        modelname = model_name.split('/')[1]
    else:
        modelname = model_name
    with open(checkpoint_dir+modelname+'_result.txt', mode='a', encoding='utf-8') as myfile:
        # for lines in text:
        myfile.write(text)
        myfile.write('\n')
    print('done')

def set_seed(seed):
    """ Set all seeds to make results reproducible (deterministic mode).
        When seed is a false-y value or not supplied, disables deterministic mode. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def write_to_file(text):
    file_path='./output/'+model_name+'_'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+mode+'_'+dataset+'_'+str(weighted_loss)

    with open(file_path+'best_model_epoch.txt', mode='a', encoding='utf-8') as myfile:

        myfile.write("epoch  "+ str(text) )
        myfile.write('\n')
    print('done')
def save_file(df,path,name):

  df.to_pickle(path+name+'.pickle')
  df.to_csv(path+name+'.csv')
  print('done')
########################prepare_data_For_model########################
def encode(docs):
  '''
  This function takes list of texts and returns input_ids and attention_mask of texts
  '''
  encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=max_length, padding='max_length',
                                             return_attention_mask=True, truncation=True, return_tensors='pt')
  input_ids = encoded_dict['input_ids']
  attention_masks = encoded_dict['attention_mask']
  return input_ids, attention_masks

def prepare_dataloader():

  train_input_ids, train_att_masks = encode(train_df[col_name].values.tolist())
  valid_input_ids, valid_att_masks = encode(valid_df['strategy'].values.tolist())
  # test_input_ids, test_att_masks = encode(test_df['strategy'].values.tolist())
  text='making the work a game'
  test_input_ids, test_att_masks = encode([text])
  # test_input_ids, test_att_masks = encode(texts)

  train_y = torch.LongTensor(train_df['label'].values.tolist())
  valid_y = torch.LongTensor(valid_df['label'].values.tolist())
  test_y = torch.LongTensor(test_df['label'].values.tolist())
  # test_y = torch.LongTensor([0]*len(texts))


  train_dataset = TensorDataset(train_input_ids, train_att_masks, train_y)
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

  valid_dataset = TensorDataset(valid_input_ids, valid_att_masks, valid_y)
  valid_sampler = SequentialSampler(valid_dataset)
  valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

  test_dataset = TensorDataset(test_input_ids, test_att_masks, test_y)
  test_sampler = SequentialSampler(test_dataset)
  test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
  return train_dataloader, valid_dataloader, test_dataloader
  # return  test_dataloader
###############initialize mode###########

def initialize_model(class_labels):


  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = BertForSequenceClassification.from_pretrained(model_name,
                                                        num_labels=class_labels,
                                                        output_attentions=False,
                                                        output_hidden_states=False)
  return  model, tokenizer
def set_label(df1):
  if mode=='binary':
    #strg. vs. no-stragy

    df1['label']= df1['label'].replace(1,0)
    df1['label']= df1['label'].replace(2,0)
    df1['label']= df1['label'].replace(3,0)
    df1['label']= df1['label'].replace(4,1)
  elif mode=='ex_ins':
      df1=df1[df1['label']!=4]
      df1['label']= df1['label'].replace(1,1)
      df1['label']= df1['label'].replace(2,1)
      df1['label']= df1['label'].replace(3,1)

  return df1

#########################compute_metrics#################################
def compute_metrics(labels, preds):
    # labels = pred.label_ids
    # preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro")
    rec = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average='macro')
    if '/' in model_name:
      modelname = model_name.split('/')[1]
    else:
      modelname = model_name

    # save_file(pd.DataFrame({'label': labels, 'pred': preds}),path,
    #           'correct_class_' + modelname + '_' + mode + '_prospect_snf')

    return {
      'accuracy': acc,
      'precision': prec,
      'recall': rec,
      'f1-score': f1
    }
def load_model_from_ckpoint(model,seed):
    print(checkpoint_dir)
    if seed>-1:
        checkpoint=checkpoint_dir+'seed'+str(seed)+'/'
    else:
        checkpoint=checkpoint_dir
    if load_model and os.path.isfile(os.path.join(checkpoint, "best_model.pth")):
      print('loading saved model from %s'%(checkpoint))

      # create new OrderedDict that does not contain `module.`
      checkpoint = torch.load(os.path.join(checkpoint, "best_model.pth"), map_location=torch.device("cpu"))
      # model.load_state_dict(new_state_dict, strict=False)
      # model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth")))

      from collections import OrderedDict

      new_state_dict = OrderedDict()
      for key, v in checkpoint.items():
          name = key.replace("module.", "")  # remove `module.`
          new_state_dict[name] = v

      # load params
      model.load_state_dict(new_state_dict, strict=False)
      return model
      # print(model)
#############################evaluate####################################

def evaluate(model, mode='validation',seed=0):
  '''
  Validation
  '''
  model.to(device)
  model.eval()
  valid_loss = 0
  valid_pred = []

  if mode == "validation":
    with torch.no_grad():
      for step_num_e, batch_data in enumerate(tqdm(valid_dataloader, desc='Validation')):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

        # loss = output.loss
        valid_loss += output[0].item()

        # valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(), axis=-1))
        valid_pred.append(np.argmax(output[1].cpu().detach().numpy(), axis=-1))

    valid_pred = np.concatenate(valid_pred)
    val_loss = valid_loss / (step_num_e + 1)
    return val_loss, np.array(valid_pred),None
  else:
    test_pred = []
    # prob={'prob_0':[],
    #       'prob_1':[],
    #       'prob_2':[],
    #       'prob_3':[],
    #       'prob_4':[]}
    prob=[]
    test_loss = 0
    if load_model:
        model=load_model_from_ckpoint(model,seed)
    with torch.no_grad():
      for step_num, batch_data in tqdm(enumerate(test_dataloader)):
        # print('step_num')
        # print(step_num)
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        # print(input_ids.shape)
        # print(labels.shape)
        # input_ids, att_mask = [data.to(device) for data in batch_data]
        # output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
        output = model(input_ids=input_ids, attention_mask=att_mask)
        # print(output)
        print('logits shape')
        print(output.logits.shape)
        # normalized = F.softmax(output[1], dim=-1).cpu().detach().numpy() #if there is label
        normalized = F.softmax(output.logits, dim=-1).cpu().detach().numpy()#nolabel
        loss = output.loss
        # test_loss += output[0].item()

        # print(output[1].shape)
        # logits=output[1].cpu().detach().numpy()
        logits = output.logits.cpu().detach().numpy()

        # test_pred.append(np.argmax(output.logits.cpu().detach().numpy(), axis=-1))
        test_pred.append(np.argmax(logits, axis=-1))
        # normalized=F.softmax(logits, dim=-1)
        # print(normalized)
        for i in range(len(normalized)):

            # for idx,probs in enumerate(normalized[i]):
            # print(i)
            # print(list(normalized[i]))
            prob.append(list(normalized[i]))

    test_pred = np.concatenate(test_pred)
    test_loss = test_loss / (step_num + 1)
    return test_loss, test_pred,prob
##############################################train##############################
def train(seed):

  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=0,
                                              num_training_steps=len(train_dataloader) * EPOCHS)
  train_loss_per_epoch = []
  val_loss_per_epoch = []
  # Set the seed value all over the place to make this reproducible.
  set_seed(seed=seed)

  best_model=None
  best_acc=0

  for epoch_num in range(EPOCHS):

      print('Epoch: ', epoch_num + 1)
      '''
      Training
      '''
      model.to(device)
      model.train()
      train_loss = 0
      for step_num, batch_data in enumerate(tqdm(train_dataloader,desc='Training')):
          input_ids, att_mask, labels = [data.to(device) for data in batch_data]

          if weighted_loss:
              output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)
              loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, requires_grad=False).to(device))
              # print(output)

              loss = loss_fct(output[1].view(-1, num_labels), labels.view(-1))
          else:
              output= model(input_ids = input_ids, attention_mask=att_mask, labels= labels)


          # print(output)
          # loss = output.loss
          train_loss += loss.item()

          model.zero_grad()
          loss.backward()
          del loss

          clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
          optimizer.step()
          scheduler.step()
      train_loss=train_loss / (step_num + 1)
      train_loss_per_epoch.append(train_loss / (step_num + 1))
      val_loss,valid_pred,_=evaluate(model, mode='validation')
      val_loss_per_epoch.append(val_loss)
      val_result=compute_metrics(valid_df.label.to_numpy(),valid_pred)
      if val_result['accuracy']>best_acc:
        print('acc improved')
        print(val_result['accuracy'])
        best_acc=val_result['accuracy']
        best_model=model
        print(f'Saving  model...')
        if not os.path.exists(checkpoint_dir+'seed'+str(seed)+'/'):
            os.makedirs(checkpoint_dir+'seed'+str(seed)+'/')
        torch.save(best_model.state_dict(), checkpoint_dir+'seed'+str(seed)+'/'+'best_model.pth')
        write_to_file(epoch_num)





      '''
      Loss message
      '''
      print("{0}/{1} train loss: {2} ".format(step_num+1, math.ceil(len(train_df) / BATCH_SIZE), train_loss))
      print("{0}/{1} val loss: {2} ".format(step_num+1, math.ceil(len(valid_df) / BATCH_SIZE), val_loss))
      if epoch_num==6:
          break
  return best_model

def get_final_output(pred, classes):
    predictions = pred[0]

    classes = np.array(classes)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)

    for i in range(pred.shape[1]):
        print("%s has confidence = %s" % (classes[i], (predictions[i]*100)))

def error_Analysis(test_df):
    ###############################contigency_matrix####################
  print(contingency_matrix(test_df['label'], test_df['pred']))
  print('printing missclassified examples')
  test_df.reset_index(level=0)
  df = test_df[test_df['label'] == test_df['pred']][['strategy', 'label', 'pred']][:50]
  print(df['strategy'].head(10))
  print(df['label'].head(10))
  print(df['pred'].head(10))
  # df=test_df[test_df['label'] != test_df['pred']][['strategy', 'label', 'pred']][:50]
  # save_file(df,checkpoint_dir,'all_hit'+mode)
  #
  # print('printing missclassified examples of non-strategy (4) class')
  # df= test_df.loc[test_df['label'] == 4]
  # df=df[df['label'] != df['pred']][['strategy', 'label', 'pred']][:100]
  # save_file(df, checkpoint_dir, 'nonstrategy_miss'+mode)



##############driver code###########

#read data
# train_df=pd.read_csv(path+'flan-t5-xl_augment_len_276_null.csv')
train_df=pd.read_csv(path+'train_extended_augmented_flant5-xl_pvi_filtered_perintent.csv')
# train_df=pd.read_csv(path+'train_forum_com_undersample_class4_1000.csv')
# print(train_df.columns.tolist())
# train_df = train_df.drop_duplicates(subset=['strategy'])
# valid_df=pd.read_csv(path+'validation_forum_binary.csv')
valid_df=pd.read_csv(path+'validation.csv')
# valid_df = valid_df.drop_duplicates(subset=['strategy'])
test_df=pd.read_csv(path+'test.csv')
# test_df=test_df.loc[test_df['strategy']=="Using tablet at school- teachers use hand over hand to show him how to do the activity."]
# test_df = test_df.drop_duplicates(subset=['strategy'])
# save_file(train_df,'./data/','train')
# save_file(valid_df,'./data/','validation')
# save_file(test_df,'./data/','test')
#   class_labels=len(set(train_df.label)
# test_result=compute_metrics(test_df.label.to_numpy(),np.array([4]*len(test_df)))
# print(test_result)
if mode=='binary' or mode=='ex_ins':
  train_df=set_label(train_df)
  valid_df=set_label(valid_df)
  test_df=set_label(test_df)
#   save_file(train_df,'train_forum_'+mode)
#   save_file(valid_df,'valid_forum_'+mode)
#   save_file(test_df,'test_forum_'+mode)
  class_labels=len(set(train_df.label))
label_list=Counter(train_df['label'])

for seed in [0]:

    if weighted_loss:
        class_weights = [len(train_df)/(label_list[i]*len(label_list)) for i in range(len(label_list))]
    # class_weights={"weight":weights}
    # #################model######################
    num_labels=len(train_df.label.unique())
    print('num_labels')
    print(num_labels)
    # num_labels = len(test_df.label.unique())
    if 'Bio_ClinicalBERT' in  model_name:
      model_name = "emilyalsentzer/Bio_ClinicalBERT"
    model, tokenizer=initialize_model(class_labels=num_labels)

    ########################get dataloader#########################
    train_dataloader, valid_dataloader, test_dataloader= prepare_dataloader()
    # test_dataloader = prepare_dataloader()

    ############################train#####################################
    if do_train:
        if load_model:
            print('load model')
            model=load_model_from_ckpoint(model)
        # model=train(seed)
    ############################predict & evaluate########################
    if do_test:

        test_loss, test_pred, prob = evaluate(model, mode="test", seed=seed)
        print(test_pred)
        test_result=compute_metrics(test_df.label.to_numpy(),test_pred)
        print(test_df['label'])
        print(test_result)

        content=str(seed)+'\t'+str(test_result['accuracy'])+'\t'+str(test_result['precision'])+'\t'+str(test_result['recall'])+'\t'+str(test_result['f1-score'])
        save_to_file(content)

        test_df['pred'] = test_pred
        test_df['prob'] = prob

        if mode=='multi':
          class_names=['EC','SOS','P','AC','n-strategy']
        elif mode=='ex/ins':
          class_names=['Extrinsic','Intrinsic']
        else:
          class_names=['strategy','no-strategy']
        # print(classification_report(test_labels, test_pprediction, target_names=class_labels))
        print(classification_report(test_df['label'], test_df['pred'], target_names=class_names))

        # if  '/' in model_name:
        #     model_name=model_name.split('/')[1]
        save_file(test_df, checkpoint_dir,model_name+'test_result_'+dataset+str(mode)+'_'+str(BATCH_SIZE)+str(weighted_loss))
        #
        # #######################evaluation################
        error_Analysis(test_df)