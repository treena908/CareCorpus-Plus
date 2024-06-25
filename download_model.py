from transformers import BertTokenizerFast, BertForSequenceClassification, BertTokenizer
model_name="bert-base-uncased"
mode="multi"
if mode=='multi':
    class_labels=5
else:
    class_labels=2
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name,
                                                        num_labels=class_labels,
                                                        output_attentions=False,
                                                        output_hidden_states=False)
if '/' in model_name:
    model_name=model_name.split('/')[1]
tokenizer.save_pretrained('./model/'+model_name+'/')
model.save_pretrained('./model/'+model_name+'/')