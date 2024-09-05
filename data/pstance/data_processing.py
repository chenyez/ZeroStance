import json
import pandas as pd
import random

train_data_path = './raw_train_all_onecol.csv'
train_data_df = pd.read_csv(train_data_path)

val_data_path = './raw_val_all_onecol.csv'
val_data_df = pd.read_csv(val_data_path)

test_data_path = './raw_test_all_onecol.csv'
test_data_df = pd.read_csv(test_data_path)

train_data_df.columns = ['Tweet','Target 1','Stance 1']
val_data_df.columns = ['Tweet','Target 1','Stance 1']
test_data_df.columns = ['Tweet','Target 1','Stance 1']

def save_csv(tosave_df,tosave_dir,trainvaltest):
	# tosave_df = pd.DataFrame(tosave_list,columns=['Tweet','Target 1','Stance 1'])
	tosave_dir = tosave_dir+'raw_'+trainvaltest+'_all_onecol.csv'

	tosave_df.to_csv(tosave_dir,index=False)
	print(tosave_dir," save, done!")

save_csv(val_data_df,'./','val')
save_csv(train_data_df,'./','train')
save_csv(test_data_df,'./','test')