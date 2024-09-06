import json
import pandas as pd
import random
random.seed(1)
# train:val:test = 10:2:3


file_path = './wtwt_dataset.csv'
file_df = pd.read_csv(file_path)

merger_set = set() 
for idx, row in file_df.iterrows():
	merger = row['merger']
	merger_set.add(merger)
print("merger_set:",merger_set)
# print(bk)
# merger	stance	text
company_pair_match = {
	'CVS_AET':'The merge of Company CVS Health and Company Aetna',
	'CI_ESRX':'The merge of Company Cigna and Company Express Scripts',
	'ANTM_CI':'The merge of Company Anthem and Company Cigna',
	'AET_HUM':'The merge of Company Aetna and Company Humana',
	'FOXA_DIS':'The merge of Company Disney and Company 21st Century Fox'
}

stance_match = {
	'support':'FAVOR',
	'refute':'AGAINST',
	'unrelated':'NONE'
}

results_train_tosave = []
results_val_tosave = []
results_test_tosave = []

for idx, row in file_df.iterrows():
	merger = row['merger']
	target = company_pair_match[merger]

	text = row['text']

	stance = row['stance']
	if stance not in stance_match.keys():
		continue

	stance = stance_match[stance]

	choice = random.randint(1,15)
	if choice in [1,2]:
		results_val_tosave.append([text,target,stance])
	elif choice in [3,4,5]:
		results_test_tosave.append([text,target,stance])
	else:
		results_train_tosave.append([text,target,stance])

print("results_train_tosave:",results_train_tosave[0],len(results_train_tosave))
print("results_val_tosave:",results_val_tosave[0],len(results_val_tosave))
print("results_test_tosave:",results_test_tosave[0],len(results_test_tosave))

	
def save_csv(tosave_list,tosave_dir,trainvaltest):
	tosave_df = pd.DataFrame(tosave_list,columns=['Tweet','Target 1','Stance 1'])
	tosave_dir = tosave_dir+'raw_'+trainvaltest+'_all_onecol.csv'

	tosave_df.to_csv(tosave_dir,index=False)
	print(tosave_dir," save, done!")

save_csv(results_val_tosave,'./','val')
save_csv(results_test_tosave,'./','test')
save_csv(results_train_tosave,'./','train')







