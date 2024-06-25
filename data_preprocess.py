import re
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
path="./data/"
def save_file(df,name):

  df.to_pickle(path+name+'.pickle')
  df.to_csv(path+name+'.csv')
  print('done')
def generate_null_df(name):

    train=pd.read_csv(path+name+'.csv')
    if 'augmented_strategy' in train.columns.tolist():
        train=train.drop(['augmented_strategy'],axis=1)
    df = pd.DataFrame({"label":train["label"],"ID":train["ID"],"strategy":train["strategy"]})
    df["augmented_strategy"]=[" "]*len(train)
    # train['null_strategy']=strategy
    save_file(df,name+"_null")
def print_stat(dfinal):
    print('PVI avrg. PVI')
    print('max %.3f min %.3f avrg %.3f median %0.3f std %0.3f'%(np.array(dfinal['PVI']).max(),np.array(dfinal['PVI']).min(),np.array(dfinal['PVI']).mean(),np.median(np.array(dfinal['PVI'])),np.array(dfinal['PVI']).std()))

def match_df(match_col=["ID","strategy"]):
    std_df=pd.read_csv(path+'flan-t5-xl_augment_vinfo'+'.csv')
    null_df = pd.read_csv(path + 'flan-t5-xl_augment_vinfo_null' + '.csv')
    print(std_df.columns.tolist())
    print(null_df.columns.tolist())
    dfinal = std_df.merge(null_df[['H_yb','predicted_label_null','correct_yb','ID','strategy']], on=match_col, how='inner')
    # print_stat(dfinal)
    dfinal['PVI']=dfinal['H_yb'] - dfinal['H_yx']
    save_file(dfinal,'bert-base-uncased_flan-t5-xl_pvi')
def remove_final_punc(strategy):
    print(strategy)
    if strategy[len(strategy)-1]=='.':
        print('yes')
        return strategy[:-1]
    return strategy
def del_particular_row(df1,df2):
    print('before %d' % (len(df1)))
    for idx,row in df2.iterrows():

        indexes=df1.index[df1.strategy == row['strategy']].tolist()
        for i in indexes:
            df1.drop(i,inplace=True)
    print('after %d' % (len(df1)))
    return df1
def clean_df(data):
    print('before %d' % (len(data)))
    data = data.dropna(axis=0, subset=['strategy'])
    clean_strategy = [remove_non_ascii(strategy) for strategy in data.strategy.values.tolist()]
    data=data.drop(['strategy'],axis=1)
    data['strategy'] = clean_strategy
    for idx,row in data.iterrows():

        if 'N/A' in row['strategy'] or len(row['strategy'])<=1 or row['strategy']=="" or row['strategy']==" ":
                try:
                    data.drop(idx, inplace=True)
                except:
                    print(idx)

    print('after %d' % (len(data)))
    return data
#clean and filter df
def fiter_df(data):
    category = ["environment/context", "sense of self", "preferences", "activity competence", "non-strategy"]
    print('before %d' % (len(data)))
    # data = data.dropna(axis=0, subset=['strategy', 'augmented_strategy'])
    data = data.dropna(axis=0, subset=['strategy'])

    data = data.drop_duplicates(subset=['augmented_strategy'])
    clean_aug_strategy = [remove_non_ascii(strategy) for strategy in data.augmented_strategy.values.tolist()]
    clean_strategy = [remove_non_ascii(strategy) for strategy in data.strategy.values.tolist()]

    data['strategy_filtered'] = clean_strategy
    data['augmented_strategy_filtered'] = clean_aug_strategy
    data = data.loc[data['strategy_filtered'] != data['augmented_strategy_filtered']]
    data.drop(['strategy_filtered', 'augmented_strategy_filtered'], axis=1)



    # for idx,row in data.iterrows():
    #     for cat in category:
    #         if (cat in row['augmented_strategy'].lower() ) or 'N/A' in row['augmented_strategy'] or 'N/A' in row['strategy']:
    #             try:
    #                 data.drop(idx, inplace=True)
    #             except:
    #                 print(idx)

    print('after %d' % (len(data)))


    # data.drop(['strategy','augmented_strategy'],axis=1)
    # data['strategy']=clean_strategy
    # data['augmented_strategy'] = clean_aug_strategy
    save_file(data,'flan-t5-xl_augment_len_276')
def concat_df():
    df = pd.read_csv('./data/train.csv')
    # df_filtered_pvi_perintent=pd.read_csv('./data/bert-base-uncased_flan-t5-xl_pvi_filtered_perintent.csv')
    df_filtered_pvi_perintent = pd.read_csv('./data/bert-base-uncased_flan-t5-xl_pvi_filtered_avrg.csv')
    # print(df['label'].value_counts())
    df_filtered_pvi_perintent.drop(['strategy'], axis=1)
    df_filtered_pvi_perintent['strategy'] = df_filtered_pvi_perintent['augmented_strategy']
    data = pd.concat(
        [df[['ID', 'strategy', 'label']], df_filtered_pvi_perintent[['ID', 'strategy', 'label']]]).reset_index()
    data = data.drop_duplicates(subset=['strategy'])
    data = shuffle(data)
    print(len(data))
    print(data.head(5))
    print(data['label'].value_counts())
    save_file(data, 'train_extended_augmented_flant5-xl_pvi_filtered_avrg')

def remove_non_ascii(text):

    text= re.sub(r'[^\x00-\x7F]+', '', text)
    return text
    # return remove_final_punc(text)
#calculate avrg pvi per class to use as threshold
def per_intent_pvi(df):
    df_class = np.array(df['PVI'].values.tolist())
    print(' PVI avrg %0.3f stdev %0.3f  max %0.3f min %0.3f med %0.3f' % (df_class.mean(), df_class.std(), df_class.max(), df_class.min(), np.median(df_class)))
    for label in [0,1,2,3,4]:
        df_class=np.array(df.loc[df['label']==label]['PVI'].values.tolist())
        print(' class %d PVI avrg %0.3f stdev %0.3f  max %0.3f min %0.3f med %0.3f'%(label, df_class.mean(), df_class.std(),df_class.max(),df_class.min(),np.median(df_class)))

    # class 0 PVI avrg -0.008 stdev 0.423  max 0.696 min -1.054 med 0.011
    #
    # class 1 PVI avrg 0.549 stdev 0.361  max 1.038 min -0.151 med 0.588
    #
    # class 2 PVI avrg 0.928 stdev 0.468  max 1.588 min 0.178 med 0.915
    #
    # class 3 PVI avrg 0.740 stdev 0.306  max 1.207 min 0.389 med 0.702
    #
    # class 4 PVI avrg 0.480 stdev 0.856  max 1.547 min -1.314 med 0.362
def len_stat(strategt_len):

  print('max strategy length in words %.3f'%(strategt_len.max()))
  print('min strategy length in words %.3f'%(strategt_len.min()))
  print('avrg strategy length in words %.3f'%(strategt_len.mean()))
  print('stdev strategy length in words %.3f'%(strategt_len.std()))
  print('median strategy length in words %.3f'%(np.median(strategt_len)))
def filter_df_per_intent_pvi(method='per_intent'):
    pvi = pd.read_csv('./data/bert-base-uncased_flan-t5-xl_pvi.csv')
    if method=='per_intent':

        pvi_class0=pvi.loc[(pvi['label']==0) & (pvi['PVI']>=0.450)]
        pvi_class1=pvi.loc[(pvi['label']==1) & (pvi['PVI']>=0.549)]
        pvi_class2=pvi.loc[(pvi['label']==2) & (pvi['PVI']>=0.928)]
        pvi_class3=pvi.loc[(pvi['label']==3 )& (pvi['PVI']>=0.740)]
        pvi_class4=pvi.loc[(pvi['label']==4) & (pvi['PVI']>=0.480)]
        classes=[pvi_class0,pvi_class1,pvi_class2,pvi_class3,pvi_class4]
        result = pd.concat(classes)
        print(len(result))
        print(result['label'].value_counts())
        save_file(result, 'bert-base-uncased_flan-t5-xl_pvi_filtered_perintent')
    else:
        result = pvi.loc[(pvi['PVI'] >= 0.188)]
        save_file(result, 'bert-base-uncased_flan-t5-xl_pvi_filtered_avrg')

data=pd.read_csv('./data/bert-base-uncased_flan-t5-xl_pvi_filtered_perintent.csv')

# print(len(df))
# forum=df['Forum'].unique()
# print(forum)
# for name in forum:
#     print(name)
#     df_forum=df.loc[df['Forum']==name]
#     print(len(df_forum))
# per_intent_pvi(df)
# concat_df()
# filter_df_per_intent_pvi(method='avrg')
# train=pd.read_csv('./data/train_forum_com_undersample_class4_1000.csv')
# forum=pd.read_csv(path+'bert-base-uncased_flan-t5-xl_pvi_filtered_avrg.csv')
# print(len(forum))
# print(df['label'].value_counts())
# forum=clean_df(forum)
# save_file(forum, 'non-strategy_forum')

# print(train_df.columns.tolist())
# train_df = train_df.drop_duplicates(subset=['strategy'])
# valid_df=pd.read_csv(path+'validation_forum_binary.csv')
# valid_df=pd.read_csv(path+'validation.csv')
# print(len(valid_df))
# # valid_df = valid_df.drop_duplicates(subset=['strategy'])
# test_df=pd.read_csv(path+'test.csv')
# print(len(test_df))

# test=pd.read_csv('./data/test.csv')
# validation=pd.read_csv('./data/validation.csv')
#
#
# filter_df_per_intent_pvi()
# df=pd.read_csv('./data/bert-base-uncased_flan-t5-xl_pvi.csv')
# df=del_particular_row(df,test)
# df=del_particular_row(df,validation)
# print(df['label'].value_counts())
# save_file(df, 'bert-base-uncased_flan-t5-xl_pvi')
# for label in [0,1,2,3,4]:
# print(label)
# data=df.loc[df['label']==label]
strategt_len=[len(strategy.split()) for strategy in data.strategy]
strategt_len=np.array(strategt_len)
len_stat(strategt_len)

# match_df()
# train=pd.read_csv('./data/flan-t5-xl_augment_len_276.csv')
# fiter_df(train)
# valid=pd.read_csv('./data/validation.csv')
# test=pd.read_csv('./data/test.csv')
# count=train['label'].value_counts()
# print(float(1486/346))# ratio class 0 class 1
# print(float(1486/271))# ratio class 0 class 2
# print(float(1486/268))# ratio class 0 class 4
# print(float(1486/166))# ratio class 0 class 3

# 4.294797687861272
# 5.483394833948339
# 5.544776119402985
# 8.951807228915662

# print(remove_non_ascii("Spend time doing activities in a day ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“ the game is still on the table, but is still challenging."))
# def clean_text(text)
# pvi_train=pd.read_csv('./data/train_extended_augmented_flant5-base_pvi_filtered_pointfive.csv')
# train=pd.read_csv('./data/train_extended_augmented_flant5-base_pvi_filtered_pointfive.csv')
# print(train.label.value_counts())
# train=pd.read_csv('./data/train.csv')
# print(train.label.value_counts())
# searchfor = ['rewrite', 'rerwitng', 'Rewrite','Rewriting']
# train = train[~train.strategy.str.contains('|'.join(searchfor))]
# print(train['label'].value_counts())
# clean_strategy=[remove_non_ascii(strategy) for strategy in train.strategy.values.tolist()]
# train['clean_strategy']=clean_strategy
# print(df_filtered_pointfive['label'].value_counts())

# df=df[df['strategy']!='N/A']
# df=df[df['strategy']!=df['augmented_strategy']]
# print('dataset len %d'%(len(df)))
# print(df['label'].value_counts())
# df_filtered_zero=df[df['PVI']>0.0]
# print('dataset len %d'%(len(df_filtered_zero)))
# print(df_filtered_zero['label'].value_counts())
# save_file(df_filtered_zero,'./data/','bert-base-uncased_flan-base_pvi_filtered_zero')
# df_filtered_pointthree=df[df['PVI']>0.3]
# print('dataset len %d'%(len(df_filtered_pointthree)))
# print(df_filtered_pointthree['label'].value_counts())
# save_file(df_filtered_pointthree,'./data/','bert-base-uncased_flan-base_pvi_filtered_pointthree')
# df_filtered_pointfive=df[df['PVI']>0.5]
# print('dataset len %d'%(len(df_filtered_pointfive)))
# print(df_filtered_pointfive['label'].value_counts())
# save_file(df_filtered_pointfive,'./data/','bert-base-uncased_flan-base_pvi_filtered_pointfive')

# match_df()
# generate_null_df('flan-t5-xl_augment_len_276')
# generate_null_df('validation')
# train=pd.read_csv(path+'flan-t5-base_augment_256_null'+'.csv')
# print(len(train.loc[13,'augmented_strategy']))
