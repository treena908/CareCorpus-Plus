from v_info import v_info

v_info(
  f"./data/flan-t5-xl_augment_len_276.csv", # data file for the augmented file where strategy is augmented
  f"./output/bert-base-uncased_16_10_multi_extended_True/checkpoint/", # the model path trained with actual strategy train file
  f"./data/flan-t5-xl_augment_len_276_null.csv", # file witk null strategy
  f"./output/bert-base-uncased_16_10_multi_extended_null_strategy_True/checkpoint/", # model path trained with null strategy
  'bert-base-uncased', #basic model name
  out_fn=f"./data/validation_pvi.csv" #output file where pvi score will be stored
)