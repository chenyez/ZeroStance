import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils.data_helper as dh
import utils.preprocessing as pp
from transformers import AdamW
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from utils import modeling, evaluation, model_utils
from sklearn.metrics import precision_recall_fscore_support,classification_report


def compute_performance(preds, y, trainvaltest, step, args, seed):

    preds_np = preds.cpu().numpy()
    y_train2_np = y.cpu().numpy()
    if len(set(y_train2_np.tolist())) == 2:
        preds_np = preds_np[:,[0,1]]
        target_names = ['Against', 'Favor']
    else:
        target_names = ['Against', 'Favor', 'Neutral']
    preds_np = np.argmax(preds_np, axis=1)
    results_weighted = precision_recall_fscore_support(y_train2_np, preds_np, average='macro')
    
    print("-------------------------------------------------------------------------------------")
    print(trainvaltest + " classification_report for step: {}".format(step))
    print(classification_report(y_train2_np, preds_np, target_names = target_names, digits = 4))
    ###############################################################################################
    ################            Precision, recall, F1 to csv                     ##################
    ###############################################################################################
    results_ind = precision_recall_fscore_support(y_train2_np, preds_np, average=None)
    results_weighted = precision_recall_fscore_support(y_train2_np, preds_np, average='macro')
    print("results_weighted:", results_weighted)
    result_overall = [results_weighted[0], results_weighted[1], results_weighted[2]]
    result_against = [results_ind[0][0], results_ind[1][0], results_ind[2][0]]
    result_favor = [results_ind[0][1], results_ind[1][1], results_ind[2][1]]
    if len(set(y_train2_np.tolist()))==2:
        result_neutral = [0, 0, 0]
    else:
        result_neutral = [results_ind[0][2], results_ind[1][2], results_ind[2][2]]

    print("result_against:", result_against)
    print("result_favor:", result_favor)
    print("result_neutral:", result_neutral)
    print("result_overall:", result_overall)

    result_id = [trainvaltest,args['dataset'], step, seed, args['dropout']]
    result_one_sample = result_id + result_against + result_favor + result_neutral + result_overall
    result_one_sample = [result_one_sample]
    print("result_combined:", result_one_sample)

    results_df = pd.DataFrame(result_one_sample)
    results_df.to_csv('./results_'+trainvaltest+'_df.csv', index=False, mode='a', header=False)    
    print('./results_'+trainvaltest+'_df.csv saved!')
    print("----------------------------------------------------------------------------")

    return results_weighted[2], result_one_sample
    
def run_classifier():

    parser = argparse.ArgumentParser()
    parser.add_argument('all_datasets', help='Datasets used in leave-one-out setting', nargs="*")
    parser.add_argument('-c', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-s', '--seed', help='Random seed', required=False)
    parser.add_argument('-d', '--dropout', help='Dropout rate', required=False)
    parser.add_argument('-train', '--train_data', help='Name of the training data file', required=False)
    parser.add_argument('-dev', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file', default=None, required=False)
    parser.add_argument('-clipgrad', '--clipgradient', type=str, default='True', help='whether clip gradient when over 2', required=False)
    parser.add_argument('-step', '--savestep', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-es_step', '--earlystopping_step', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-dataset', '--dataset', help='dataset name', default="vast", required=False)
    parser.add_argument('-leave_one_out', '--leave_one_out', type=int, default=1, help='0: in domain, 1: leave one out', required=False)
    parser.add_argument('-lr1', '--lr1', type=float, default=1e-5, help='lr for the main model', required=False)
    parser.add_argument('-lr2', '--lr2', type=float, default=1e-5, help='lr for the output layer', required=True)

    args = vars(parser.parse_args())
    num_labels = 3  # Favor, Against and None
    random_seeds = []
    random_seeds.append(int(args['seed']))
    
    # create normalization dictionary for preprocessing
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    norm_dict = {**data1,**data2}
    
    # load config file
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    
    config['bert_lr'] = args['lr1']
    config['fc_lr'] = args['lr2']
    all_datasets = args['all_datasets']
    target_dataset = args['dataset']
    print("PLM lr:", config['bert_lr'])
    print("FC lr:", config['fc_lr'])
    print("Batch size:", config['batch_size'])
    print("Dropout:", str(args['dropout']))
    print("Early stopping patience:", str(args['earlystopping_step']))
    print("Clip gradient:", args['clipgradient'])
    print("Datasets used in leave-one-out setting:", args['all_datasets'])
    print(60*"#")
    model_select = config['model_select']
    
    # Use GPU or not
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    for seed in random_seeds:    
        print("Current random seed: ", seed)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+'_dropout'+str(args['dropout'])+'_seed'+str(seed), 'train')
        train_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+'_dropout'+str(args['dropout'])+'_seed'+str(seed), 'val')
        val_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+'_dropout'+str(args['dropout'])+'_seed'+str(seed), 'test')
        test_writer = SummaryWriter(log_dir=log_dir)

        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        
        #############################################################################
        # for train set, combine all datasets
#         all_datasets = ["vast", "perspectrum", "ibm30k", "covid19","semeval2016", "argmin", "wtwt", "pstance"]
#         all_datasets = ["vast", "ibm30k", "covid19","semeval2016", "wtwt", "pstance"]

        # train set is to simply concat train data of all datasets
        x_train, y_train, x_train_target = [], [], []
        
        # val set is to simply concat train data of all datasets
        x_val, y_val, x_val_target = [], [], []
        
        # test set is the test set of the leave-one-out dataset
        x_test, y_test, x_test_target = [], [], []
        
        # for val/test set, load each dataset seperately
        if args['leave_one_out'] == 1:
            all_datasets = [x for x in all_datasets if x != args['dataset']]
            sep_val_points = [0]
            
            # load train, val dataset for all datasets
            for i in range(len(all_datasets)):
                d_name = all_datasets[i]
                train_data_path = os.path.join(args['train_data'], d_name, 'raw_train_all_onecol.csv')
                x_train_tmp, y_train_tmp, x_train_target_tmp = pp.clean_all(train_data_path,norm_dict,d_name,args['dataset']) 
                x_train += x_train_tmp
                y_train += y_train_tmp
                x_train_target += x_train_target_tmp
                
                # train_data_path = os.path.join(args['train_data'],d_name,'raw_train_all_onecol_bt.csv')
                # x_train_tmp, y_train_tmp, x_train_target_tmp = pp.clean_all(train_data_path,norm_dict,d_name,args['dataset']) 
                # x_train += x_train_tmp
                # y_train += y_train_tmp
                # x_train_target += x_train_target_tmp

                val_data_path = os.path.join(args['dev_data'], d_name, 'raw_val_all_onecol.csv')
                x_val_tmp, y_val_tmp, x_val_target_tmp = pp.clean_all(val_data_path, norm_dict, d_name, args['dataset'])
                x_val += x_val_tmp
                y_val += y_val_tmp
                x_val_target += x_val_target_tmp
                sep_val_points.append(len(y_val))
            print("val split on the following points: ", sep_val_points)

            # load the leave-one-out test set
            test_data_path = os.path.join(args['test_data'], args['dataset'], 'raw_test_all_onecol.csv')
            x_test, y_test, x_test_target = pp.clean_all(test_data_path, norm_dict, args['dataset'], args['dataset'])
        else:
            train_data_path = os.path.join(args['train_data'], args['dataset'], 'raw_train_all_onecol.csv')
            x_train, y_train, x_train_target = pp.clean_all(train_data_path, norm_dict, args['dataset'], args['dataset']) 
            val_data_path = os.path.join(args['dev_data'], args['dataset'], 'raw_val_all_onecol.csv')
            x_val, y_val, x_val_target = pp.clean_all(val_data_path, norm_dict, args['dataset'], args['dataset'])
            test_data_path = os.path.join(args['test_data'], args['dataset'], 'raw_test_all_onecol.csv')
            x_test, y_test, x_test_target = pp.clean_all(test_data_path, norm_dict, args['dataset'], args['dataset'])
        #############################################################################

        print(60*"#")
        print("Size of train set:", len(x_train))
        print("Size of val set:", len(x_val))
        print("Size of test set:", len(x_test))

        x_train_all = [x_train, y_train, x_train_target]
        x_val_all = [x_val, y_val, x_val_target]
        x_test_all = [x_test, y_test, x_test_target]

        loader, gt_label = dh.data_helper_bert(x_train_all, x_val_all, x_test_all, model_select, config)
        trainloader, valloader, testloader, trainloader2 = loader[0], loader[1], loader[2], loader[3]
        y_train, y_val, y_test, y_train2 = gt_label[0], gt_label[1], gt_label[2], gt_label[3]
        y_val, y_test, y_train2 = y_val.to(device), y_test.to(device), y_train2.to(device)       
        
        # train setup
        model, optimizer = model_utils.model_setup(num_labels, model_select, device, config, float(args['dropout']))
        print("Train setup has finished!")
        loss_function = nn.CrossEntropyLoss()
        sum_loss = []

        # early stopping
        es_intermediate_step = len(trainloader)//args['savestep']
        patience = args['earlystopping_step']   # the number of iterations that loss does not further decrease    
        early_stopping = EarlyStopping(patience, args['dataset'], args['seed'],verbose=True)
        print("Early stopping occurs when the loss does not decrease after {} steps.".format(patience))
        
        # init best val/test results
        best_train_f1macro, best_val_f1macro, best_test_f1macro = -1, -1, -1
        best_train_result, best_val_result, best_test_result = [], [], []
        best_val_loss, best_test_loss = 100000, 100000
        best_val_loss_result, best_test_loss_result = [], []
        step = 0
        # start training
        for epoch in range(0, int(config['total_epochs'])):
            print(60*"#")
            print('Epoch:', epoch)
            train_loss = []  
            model.train()
            for b_id, sample_batch in enumerate(trainloader):
                model.train()
                optimizer.zero_grad()
                dict_batch = model_utils.batch_fn(sample_batch)
                inputs = {k: v.to(device) for k, v in dict_batch.items()}
                outputs = model(**inputs)
                loss = loss_function(outputs, inputs['gt_label'])
                loss.backward()

                if args['clipgradient']=='True':
                    nn.utils.clip_grad_norm_(model.parameters(), 2)

                optimizer.step()
                step+=1
                train_loss.append(loss.item())

                split_step = len(trainloader)//args['savestep']

                if step%split_step==0:
                    model.eval()
                    with torch.no_grad():
                        preds_train, loss_train = model_utils.model_preds(trainloader2, model, device, loss_function)
                        preds_val, loss_val = model_utils.model_preds(valloader, model, device, loss_function)
                        preds_test, loss_test = model_utils.model_preds(testloader, model, device, loss_function)
                        print("At step: {}".format(step))
                        print("Train loss: ",sum(loss_train)/len(loss_train))
                        print("Val loss: ",sum(loss_val) / len(loss_val))
                        print("Test loss: ",sum(loss_test) / len(loss_test))

                        train_writer.add_scalar('loss', sum(loss_train)/len(loss_train), step)
                        val_writer.add_scalar('loss', sum(loss_val) / len(loss_val), step)
                        test_writer.add_scalar('loss', sum(loss_test) / len(loss_test), step)

                        f1macro_train, result_one_sample_train = compute_performance(preds_train,y_train2,'training',step, args, seed)
                        f1macro_val, result_one_sample_val = compute_performance(preds_val,y_val,'validation',step, args, seed)
                        f1macro_test, result_one_sample_test = compute_performance(preds_test,y_test,'test',step, args, seed)
                        
                        train_writer.add_scalar('f1macro', f1macro_train, step)
                        val_writer.add_scalar('f1macro', f1macro_val, step)
                        test_writer.add_scalar('f1macro', f1macro_test, step)
                        
                        avg_val_loss = sum(loss_val) / len(loss_val)
                        avg_test_loss = sum(loss_test) / len(loss_test)
                        
                        if f1macro_val>best_val_f1macro:
                            best_val_f1macro = f1macro_val
                            best_val_result = result_one_sample_val
                            print("Best validation result (according to f1-macro) is updated at epoch {}, as: {}".format(epoch, best_val_f1macro))
                            best_test_f1macro = f1macro_test
                            best_test_result = result_one_sample_test
                            print("Best test result (according to f1-macro) is updated at epoch {}, as: {}".format(epoch, best_test_f1macro))

                        if avg_val_loss<best_val_loss:
                            best_val_loss = avg_val_loss
                            best_val_loss_result = result_one_sample_val
                            print("Best validation result (according to loss) is updated at epoch {}, as: {}".format(epoch, best_val_loss))
                            best_test_loss = avg_test_loss
                            best_test_loss_result = result_one_sample_test
                            print("Best test result (according to loss) is updated at epoch {}, as: {}".format(epoch, best_test_loss))
                            if args['leave_one_out'] == 0:
                                model_weight = model_select+'_seed{}.pt'.format(seed)
                                torch.save(model.state_dict(), model_weight)                             
                            print(60*"#")

                        # early stopping
#                         print("loss_val:",loss_val,"average is: ",sum(loss_val) / len(loss_val))
                        early_stopping(sum(loss_val) / len(loss_val), model)
                        if early_stopping.early_stop:
                            print("Early stopping occurs at step: {}, stop training.".format(step))
                            break
                    model.train()

            if early_stopping.early_stop:
                print("Early stopping, training ends")
                print(60*"#")
                break

#             sum_loss.append(sum(train_loss)/len(train_loss))
#             print(sum_loss[epoch])

        #########################################################
        best_val_result[0][0]='best validation'        
        results_df = pd.DataFrame(best_val_result)    
#         print("results_df are:",results_df.head())
        results_df.to_csv('./results_validation_df.csv', index=False, mode='a', header=False)    
#         print('./results_validation_df.csv saved!')
        ###
        results_df = pd.DataFrame(best_val_result)    
        print("best_val_result is: ",results_df.head())
        results_df.to_csv('./best_results_validation_df.csv', index=False, mode='a', header=False)    
        print('./best_results_validation_df.csv saved!')
        ###
        best_val_loss_result[0][0]='best validation' 
        results_df = pd.DataFrame(best_val_loss_result)    
        print("best_val_loss_result is: ",results_df.head())
        results_df.to_csv('./best_loss_results_validation_df.csv', index=False, mode='a', header=False)    
        print('./best_loss_results_validation_df.csv saved!')
        #########################################################
        best_test_result[0][0]='best test'
        results_df = pd.DataFrame(best_test_result)    
#         print("results_df are:",results_df.head())
        results_df.to_csv('./results_test_df.csv', index=False, mode='a', header=False)    
#         print('./results_test_df.csv saved!')
        ###
        results_df = pd.DataFrame(best_test_result)    
        print("best_test_result is: ",results_df.head())
        results_df.to_csv('./best_results_test_df.csv', index=False, mode='a', header=False)    
        print('./best_results_test_df.csv saved!')
        ###
        best_test_loss_result[0][0]='best test'
        results_df = pd.DataFrame(best_test_loss_result)    
        print("best_test_loss_result is: ",results_df.head())
        results_df.to_csv('./best_loss_results_test_df.csv', index=False, mode='a', header=False)    
        print('./best_loss_results_test_df.csv saved!')
        
        if args['leave_one_out'] == 0:
            print("Now test the best model on separate stance datasets!")
            weight = model_select+'_seed{}.pt'.format(seed)
            model.load_state_dict(torch.load(weight))
            model.eval()
            with torch.no_grad():
                preds_test, loss_test = model_utils.model_preds(testloader, model, device, loss_function)
                preds_test_list = model_utils.sep_test_set(preds_test)
                y_test_list = model_utils.sep_test_set(y_test)
                for ind in range(len(y_test_list)):
                    args['dataset'] = all_datasets[ind] # pay attention to all_datasets defined above
                    _,result_one_sample_test = compute_performance(preds_test_list[ind], y_test_list[ind], 'test', step, args, seed)
                    best_test_loss_result = result_one_sample_test
                    best_test_loss_result[0][0]='best test'
                    results_df = pd.DataFrame(best_test_loss_result)    
                    print("best_test_loss_result of {} is: {}".format(args['dataset'], results_df.head()))
                    results_df.to_csv('./best_loss_results_test_df.csv', index=False, mode='a', header=False)    
                print('./best_loss_results_test_df.csv saved!')
         
        print(60*"#")
        
if __name__ == "__main__":
    run_classifier()
