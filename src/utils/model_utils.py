import torch
from transformers import AdamW
from utils import modeling

def model_setup(num_labels, model_select, device, config, dropout):
    
#     print("current dropout is: ", dropout)
    if model_select == 'Bert':
        print("BERT is used as the stance classifier.")
        model = modeling.bert_classifier(num_labels, model_select, dropout).to(device)
        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
        
    elif model_select == 'Bart_encoder':
        print("Bart-large-mnli encoder is used as the stance classifier.")
        model = modeling.bart_classifier(num_labels, model_select, dropout).to(device)
        for n, p in model.named_parameters():
            if "bart.shared.weight" in n or "bart.encoder.embed" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bart.encoder.layer')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
        
    elif model_select == 'RoBERTa':
        print("BERTweet is used as the stance classifier.")
        model = modeling.roberta_large_classifier(num_labels, model_select, dropout).to(device)      
        for n, p in model.named_parameters():
            if "roberta.embeddings" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('roberta')] , 'lr': float(config['bert_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ] 
        
    optimizer = AdamW(optimizer_grouped_parameters)
    
    return model, optimizer


def model_preds(loaders, model, device, loss_function):
    
    preds = [] 
    valtest_loss = []   
    for b_id, sample_batch in enumerate(loaders):
        dict_batch = batch_fn(sample_batch)
        inputs = {k: v.to(device) for k, v in dict_batch.items()}
        outputs = model(**inputs)
        preds.append(outputs)

        loss = loss_function(outputs, inputs['gt_label'])
        valtest_loss.append(loss.item())

    return torch.cat(preds, 0), valtest_loss


def batch_fn(sample_batch):
    
    dict_batch = {}
    dict_batch['input_ids'] = sample_batch[0]
    dict_batch['attention_mask'] = sample_batch[1]
    dict_batch['gt_label'] = sample_batch[-1]
    if len(sample_batch) > 3:
        dict_batch['token_type_ids'] = sample_batch[-2]
    
    return dict_batch

def sep_test_set(input_data):

    # in the order of "vast", "ibm30k", "covid19","semeval2016", "wtwt", "pstance"
    data_list = [input_data[:1460], input_data[1460:7775], input_data[7775:8575], input_data[8575:9824],
                 input_data[9824:12861], input_data[12861:15018]]

    return data_list

def sep_val_set(input_data, sep_val_points):
    
    data_list = []
    for i in range(len(sep_val_points)):
        if i < len(sep_val_points)-1:
            data_list.append(input_data[sep_val_points[i]:sep_val_points[i+1]])

    return data_list