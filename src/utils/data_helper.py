import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, AlbertTokenizer, BartTokenizer, RobertaTokenizer, BertweetTokenizer
transformers.logging.set_verbosity_error()
    

# Tokenization
def convert_data_to_ids(tokenizer, target, text, label, config):
    
    concat_sent = []
    for tar, sent in zip(target, text):
        concat_sent.append([' '.join(sent), ' '.join(tar)])
    encoded_dict = tokenizer.batch_encode_plus(
                    concat_sent,
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = int(config['max_tok_len']), # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,   # Construct attn. masks.
                    truncation = True,
               )
    encoded_dict['gt_label'] = label
    
    return encoded_dict


# BERT/BERTweet tokenizer    
def data_helper_bert(x_train_all,x_val_all,x_test_all,model_select,config):
    
    print('Loading data')
    
    x_train, y_train, x_train_target = x_train_all[0], x_train_all[1], x_train_all[2]
    x_val, y_val, x_val_target = x_val_all[0], x_val_all[1], x_val_all[2]
    x_test, y_test, x_test_target = x_test_all[0], x_test_all[1], x_test_all[2]
    
    # get the tokenizer
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    elif 'Bart' in model_select:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli", local_files_only=True)
    elif model_select == 'Bert':
#         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         tokenizer = BertTokenizer.from_pretrained("../../model_hub/covid-twitter-bert-v2")
        tokenizer = AutoTokenizer.from_pretrained("../../model_hub/twhin-bert-large")
    elif model_select=='RoBERTa':
        tokenizer = AutoTokenizer.from_pretrained('../../model_hub/bertweet-large', local_files_only=True)
        # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", local_files_only=True)
    elif model_select=='RoBERTa_mnli':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')

    # tokenization
    train_encoded_dict = convert_data_to_ids(tokenizer, x_train_target, x_train, y_train, config)
    val_encoded_dict = convert_data_to_ids(tokenizer, x_val_target, x_val, y_val, config)
    test_encoded_dict = convert_data_to_ids(tokenizer, x_test_target, x_test, y_test, config)
    trainloader, y_train = data_loader(train_encoded_dict, int(config['batch_size']), model_select, 'train')
    valloader, y_val = data_loader(val_encoded_dict, int(config['batch_size']), model_select, 'val')
    testloader, y_test = data_loader(test_encoded_dict, int(config['batch_size']), model_select, 'test')
    trainloader2, y_train2 = data_loader(train_encoded_dict, int(config['batch_size']), model_select, 'train2')
    
    return (trainloader, valloader, testloader, trainloader2), (y_train, y_val, y_test, y_train2)


def data_loader(x_all, batch_size, model_select, mode):
    
    x_input_ids = torch.tensor(x_all['input_ids'], dtype=torch.long)
    x_atten_masks = torch.tensor(x_all['attention_mask'], dtype=torch.long)
    y = torch.tensor(x_all['gt_label'], dtype=torch.long)
    if model_select == 'Bert':
#         x_seg_ids = torch.tensor(x_all['token_type_ids'], dtype=torch.long)
#         tensor_loader = TensorDataset(x_input_ids, x_atten_masks, x_seg_ids, y)
        tensor_loader = TensorDataset(x_input_ids, x_atten_masks, y)
    else:
        tensor_loader = TensorDataset(x_input_ids, x_atten_masks, y)

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

    return data_loader, y
