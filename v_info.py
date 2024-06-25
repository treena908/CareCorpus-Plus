from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, \
    BertForSequenceClassification
from tokenizers.pre_tokenizers import Whitespace
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
parser = argparse.ArgumentParser()
def save_to_file(text,name):
    with open('./data/'+name+'.txt', mode='a', encoding='utf-8') as myfile:
        # for lines in text:
        myfile.write(text )
        myfile.write('\n')
    print('done')
def get_model_tokenizer(checkpoint_dir,tokenizer):
    model = BertForSequenceClassification.from_pretrained(tokenizer,
                                                          num_labels=5,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_model.pth"), map_location=torch.device("cpu"))
    # model.load_state_dict(new_state_dict, strict=False)
    # model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth")))

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for key, v in checkpoint.items():
        name = key.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict, strict=False)
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    return model,tokenizer
def v_entropy(data_fn, model, tokenizer, input_key='sentence1', batch_size=1):
    """
    Calculate the V-entropy (in bits) on the data given in data_fn. This can be
    used to calculate both the V-entropy and conditional V-entropy (for the
    former, the input column would only have null data and the model would be
    trained on this null data).

    Args:
        data_fn: path to data; should contain the label in the 'label' column
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn
        batch_size: data batch_size

    Returns:
        Tuple of (V-entropies, correctness of predictions, predicted labels).
        Each is a List of n entries (n = number of examples in data_fn).
    """
    # added for gpt2 
    if tokenizer == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)
    else:
        model,tokenizer=get_model_tokenizer(model,tokenizer)
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True, device=0)
    data = pd.read_csv(data_fn)
    # if 'null' in data_fn:
    #     file_name="flan-t5-xl_augment_vinfo_null"
    # else:
    #     file_name = "flan-t5-xl_augment_vinfo"
    if 'null' in data_fn:
        file_name="valid_vinfo_null"
    else:
        file_name = "valid_vinfo"

    
    entropies = []
    correct = []
    predicted_labels = []
    count=0

    for j in tqdm(range(0, len(data), batch_size)):
        batch = data[j:j+batch_size]
        # print(j)
        # print(batch[input_key].tolist())
        try:
            predictions = classifier(batch[input_key].tolist())
            print("ekhane")
            # print(type(batch))
            # print(len(batch))
            # print(type(predictions))
            # print(len(predictions))

            for i in range(len(batch)):
                # print('batch')
                # print(int(batch.iloc[i]['label']))
                for d in predictions[i]:
                    #
                    # print('pred')
                    # # print(d)
                    # print(int(d['label'].split('_')[1]))

                    if int(d['label'].split('_')[1])== int(batch.iloc[i]['label']):
                        # print('match')
                        prob=d['score']
                        break


                # prob = next(d for d in predictions[i] if d['label'] == batch.iloc[i]['label'])['score']
                entropy=-1 * np.log2(prob)
                entropies.append(entropy)
                # print(predictions[i])
                predicted_label = max(predictions[i], key=lambda x: x['score'])['label'].split('_')[1]
                # print(predicted_label)
                # print(type(predicted_label))

                predicted_labels.append(predicted_label)
                correct.append(int(predicted_label) == int(batch.iloc[i]['label']))
                content = str(batch.iloc[i]['ID']) + '\t' + str(batch.iloc[i]['label']) + '\t'+ batch.iloc[i]['strategy']+ '\t'+batch.iloc[i]['augmented_strategy']+'\t'+str(entropy)+'\t'+str(predicted_label)+'\t'+str(int(predicted_label) == int(batch.iloc[i]['label']))
                print('complete')
                save_to_file(content,file_name)


        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
            # sys.exit()
            for i in range(len(batch)):
                print('error')
                count+=1
                entropies.append(-1000)
                predicted_labels.append(-1)
                correct.append(-1)
                content = str(batch.iloc[i]['ID']) + '\t' + str(batch.iloc[i]['label']) + '\t'+ str(batch.iloc[i]['strategy'])+ '\t'+str(batch.iloc[i]['augmented_strategy'])+ '\t' + str('no') + '\t' + str('no') + '\t' + str('no')


                save_to_file(content,file_name)


    torch.cuda.empty_cache()
    print('not processed %0.3f'%((count/len(data))*100))
    print('data len %d'%len(data))
    print('entropy pred corect len %d %d %d'%(len(entropies),len(predicted_labels),len(correct)))
    return entropies, correct, predicted_labels


def v_info(data_fn, model, null_data_fn, null_model, tokenizer, out_fn="", input_key='augmented_strategy'):
    """
    Calculate the V-entropy, conditional V-entropy, and V-information on the
    data in data_fn. Add these columns to the data in data_fn and return as a 
    pandas DataFrame. This means that each row will contain the (pointwise
    V-entropy, pointwise conditional V-entropy, and pointwise V-info (PVI)). By
    taking the average over all the rows, you can get the V-entropy, conditional
    V-entropy, and V-info respectively.

    Args:
        data_fn: path to data; should contain the label in the 'label' column 
            and X in column specified by input_key
        model: path to saved model or model name in HuggingFace library
        null_data: path to null data (column specified by input_key should have
            null data)
        null_model: path to saved model trained on null data
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        out_fn: where to saved 
        input_key: column name of X variable in data_fn 

    Returns:
        Pandas DataFrame of the data in data_fn, with the three additional 
        columns specified above.
    """
    data = pd.read_csv(data_fn)

    data['H_yx'], data['correct_yx'], data['predicted_label'] = v_entropy(data_fn, model, tokenizer, input_key=input_key)
    data['H_yb'], _, _ = v_entropy(null_data_fn, null_model, tokenizer, input_key=input_key)
    data['PVI'] = data['H_yb'] - data['H_yx']
    print(data['PVI'])
    if out_fn:
        data.to_csv(out_fn)

    return data


def find_annotation_artefacts(data_fn, model, tokenizer, input_key='sentence1', min_freq=5, pre_tokenize=True):
    """
    Find token-level annotation artefacts (i.e., tokens that once removed, lead to the
    greatest decrease in PVI for each class).

    Args:
        data_fn: path to data; should contain the label in the 'label' column 
            and X in column specified by input_key
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn 
        min_freq: minimum number of times a token needs to appear (in a given class' examples)
            to be considered a potential partefact
        pre_tokenize: if True, do not consider subword-level tokens (each word is a token)

    Returns:
        A pandas DataFrame with one column for each unique label and one row for each token.
        The value of the entry is the entropy delta (i.e., the drop in PVI for that class if that
        token is removed from the input). If the token does meet the min_freq threshold, then the
        entry is empty.
    """
    data = pd.read_csv(data_fn)
    labels = [ l for l in data['label'].unique().tolist() if l >= 0 ] # assume labels are numbers
    token_entropy_deltas = { l : {} for l in labels }
    all_tokens = set([])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    pre_tokenizer = Whitespace()

    # added for gpt2 
    if tokenizer == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)

    # get the PVI for each example
    print("Getting conditional V-entropies ...")
    entropies, _ = v_entropy(data_fn, model, tokenizer, input_key=input_key)

    print("Calculating token-wise delta for conditional entropies ...")
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True, device=0)

    for i in tqdm(range(len(data))):
        example = data.iloc[i]

	    # mislabelled examples; ignore these
        if example['label'] < 0:
            continue

        if pre_tokenize:
            tokens = [ t[0] for t in pre_tokenizer.pre_tokenize_str(example['sentence1']) ]
        else:
            tokens = tokenizer.tokenize(example['sentence1'])

        # create m versions of the input in which one of the m tokens it contains is omitted
        batch = pd.concat([ example ] * len(tokens), axis=1).transpose()
         
        for j in range(len(tokens)):
            # create new input by omitting token j
            batch.iloc[j][input_key] = tokenizer.convert_tokens_to_string(tokens[:j] + tokens[j+1:])
            all_tokens.add(tokens[j])

            for label in labels:
                if tokens[j] not in token_entropy_deltas[label]: token_entropy_deltas[label][tokens[j]] = []

        # get the predictions (in rare cases, have to split the batch up into mini-batches of 100 because it's too large)
        predictions = []
        for k in range(0, len(batch), 100):
            predictions.extend(classifier(batch[input_key][k:k+100].tolist()))
        
        for j in range(len(tokens)):
            prob = next(d for d in predictions[j] if d['label'] == example['label'])['score']
            entropy_delta = (-1 * np.log2(prob)) - entropies[i]
            token_entropy_deltas[example['label']][tokens[j]].append(entropy_delta)

    torch.cuda.empty_cache()

    total_freq = { t : sum(len(token_entropy_deltas[l][t]) for l in labels) for t in all_tokens }
    # average over all instances of token in class
    for label in labels:
        for token in token_entropy_deltas[label]:
            if total_freq[token] > min_freq:
            	token_entropy_deltas[label][token] = np.nanmean(token_entropy_deltas[label][token]) 
            else:
                token_entropy_deltas[label][token] = np.nan

    table = pd.DataFrame.from_dict(token_entropy_deltas)
    return table


if __name__ == "__main__":
    os.makedirs('PVI', exist_ok=True)

    parser.add_argument('--data_dir', help='raw_data directory', required=True, type=str)
    parser.add_argument('--model_dir', help='model directory', required=True, type=str)
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    MODEL_DIR = args.model_dir

    for tokenizer in ['bert-base-cased', 'facebook/bart-base', 'gpt2', 'distilbert-base-uncased', 'roberta-large']:
        model_name = tokenizer.replace('/', '-')

        # MultiNLI
        for suffix in ['validation']:
            print(model_name, 'multinli', suffix)
            v_info(f"data/multinli_{suffix}_std.csv", f"{MODEL_DIR}/{model_name}_multinli_std", f"data/multinli_{suffix}_null.csv",
                f"{MODEL_DIR}/{model_name}_multinli_null", tokenizer, out_fn=f"PVI/{model_name}_multinli_{suffix}_std.csv")

        # CoLA
        for suffix in ['train', 'id_dev']:
            for epoch in [1,2,3,5]:
                print(model_name, 'cola', suffix)
                v_info(f"data/cola_{suffix}_std.csv", f"{MODEL_DIR}/{model_name}_cola_std{epoch}",
                  f"data/cola_{suffix}_null.csv", f"{MODEL_DIR}/{model_name}_cola_null", tokenizer,
                  out_fn=f"PVI/{model_name}_cola_{suffix}_std{epoch}.csv")
       
        # DWMW
        for experiment in ['sentiment', 'std', 'bad_vocab']:
            print(model_name, 'dwmw', experiment)
            v_info(f"data/dwmw_{experiment}.csv", f"{MODEL_DIR}/{model_name}_dwmw_{experiment}", f"data/dwmw_null.csv", f"{MODEL_DIR}/{model_name}_dwmw_null",
                    tokenizer, out_fn=f"PVI/{model_name}_dwmw_{experiment}.csv")

        v_info(f"data/dwmw_sentiment_vocab.csv", f"{MODEL_DIR}/{model_name}_dwmw_sentiment_vocab", f"data/dwmw_sentiment.csv", f"{MODEL_DIR}/{model_name}_dwmw_sentiment",
            tokenizer, out_fn=f"PVI/{model_name}_dwmw_sentiment_vocab.csv")
        
        # SNLI
        experiments = [
            ("snli_train_std.csv", "snli_std"),
            ("snli_train_std.csv", "snli_std2"),
            ("snli_train_std.csv", "snli_std3"),
            ("snli_train_std.csv", "snli_std5"),
            ("snli_train_std.csv", "snli_std10"),
        ]

        for side_data, side_model in experiments:
            print(model_name, side_model)
            v_info(f"data/{side_data}", f"{MODEL_DIR}/{model_name}_{side_model}",
		f"data/snli_train_null.csv", f"{MODEL_DIR}/{model_name}_snli_null", tokenizer,
                out_fn=f"PVI/{model_name}_{side_model}_train.csv")
      
        experiments = [
            ("snli_test_std.csv", "snli_std"),
            ("snli_test_std.csv", "snli_std2"),
            ("snli_test_std.csv", "snli_std3"),
            ("snli_test_std.csv", "snli_std5"),
            ("snli_test_std.csv", "snli_std10"),
            ("snli_test_premise.csv", "snli_premise"),
            ("snli_test_hypothesis.csv", "snli_hypothesis"),
            ("snli_test_shuffled.csv", "snli_shuffled"),
        ]

        for side_data, side_model in experiments:
            print(model_name, side_model)
            v_info(f"data/{side_data}", f"{MODEL_DIR}/{model_name}_{side_model}", f"data/snli_test_null.csv", f"{MODEL_DIR}/{model_name}_snli_null", tokenizer,
                out_fn=f"PVI/{model_name}_{side_model}_test.csv")

        # fractional training
        for size in [ 0.05, 0.2, 0.4, 0.6, 0.8, 0.99 ]:
            for version in ['b', 'c', 'd', 'e']:
                print(model_name, 'snli', size)
                v_info(f"data/snli_test_std.csv", f"{MODEL_DIR}/{model_name}_snli_std_{version}_{size}",
                      f"data/snli_test_null.csv", f"{MODEL_DIR}/{model_name}_snli_null_{version}_{size}", tokenizer,
                      out_fn=f"PVI/{model_name}_snli_std_test_{version}_{size}.csv")
