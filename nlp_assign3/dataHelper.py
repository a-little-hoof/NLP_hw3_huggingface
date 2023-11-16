from datasets import Dataset,load_dataset,concatenate_datasets,ClassLabel
import datasets
import json
import os
import argparse
import random
import numpy as np


def get_dataset(dataset_name, sep_token):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
	dataset = None

	# your code for preparing the dataset...
	flag = False
	train_list = []
	test_list = []
	cnt = 0
	if dataset_name[0]=="[":
		flag=True
	
	if dataset_name.find("restaurant")!=-1:
		data_dict = {}
		for data_file in os.listdir(r"./SemEval14-res"):
			if data_file.find(".json")!=-1:
				with open("./SemEval14-res/"+data_file) as f:
					data_dict.update(json.load(f)) 

		unsplit_dict = {"text":[],"label":[]}
		for value in data_dict.values():
			unsplit_dict["text"].append(value["term"]+" "+sep_token+" "+value["sentence"])
			unsplit_dict["label"].append(value["polarity"])
		unsplit_set = Dataset.from_dict(unsplit_dict)
		def maptoidx(example):
				if example["label"]=='positive':
					example["label"] = cnt 
				if example["label"]=='neutral':
					example["label"] = cnt+1
				if example["label"]=='negative':
					example["label"] = cnt+2
				return example
		unsplit_set = unsplit_set.map(maptoidx)
		dataset = unsplit_set.train_test_split(test_size=0.2, shuffle=True)
		cnt+=3
		# print(dataset["train"][0])
		
		if flag:
			train_list.append(dataset["train"])
			test_list.append(dataset["test"])
	
	if dataset_name.find("laptop")!=-1:
		data_dict = {}
		for data_file in os.listdir(r"./SemEval14-laptop"):
			if data_file.find(".json")!=-1:
				with open("./SemEval14-laptop/"+data_file) as f:
					data_dict.update(json.load(f)) 

		unsplit_dict = {"text":[],"label":[]}
		for value in data_dict.values():
			unsplit_dict["text"].append(value["term"]+" "+sep_token+" "+value["sentence"])
			unsplit_dict["label"].append(value["polarity"])
		unsplit_set = Dataset.from_dict(unsplit_dict)
		def maptoidx(example):
				if example["label"]=='positive':
					example["label"] = cnt 
				if example["label"]=='neutral':
					example["label"] = cnt+1
				if example["label"]=='negative':
					example["label"] = cnt+2
				return example
		unsplit_set = unsplit_set.map(maptoidx)
		dataset = unsplit_set.train_test_split(test_size=0.2, shuffle=True)
		cnt+=3

		if flag:
			train_list.append(dataset["train"])
			test_list.append(dataset["test"])


	if dataset_name.find("acl")!=-1:
		data_lst = []
		for data_file in os.listdir(r"./ACL-ARC/ACL-ARC"):
			if data_file.find("json")!=-1:
				with open("./ACL-ARC/ACL-ARC/"+data_file,'rb') as f:
					lines = f.readlines()
					for line in lines:
						# print(line)
						# print(json.loads(line))
						data_lst.append(json.loads(line)) 

		unsplit_dict = {"text":[],"label":[]}
		for value in data_lst:
			unsplit_dict["text"].append(value["string"])
			unsplit_dict["label"].append(value["label"])
		unsplit_set = Dataset.from_dict(unsplit_dict)
		def maptoidx(example):
				if example["label"]=='background':
					example["label"] = cnt 
				if example["label"]=='method':
					example["label"] = cnt+1
				if example["label"]=='result':
					example["label"] = cnt+2
				return example
		unsplit_set = unsplit_set.map(maptoidx)
		dataset = unsplit_set.train_test_split(test_size=0.2, shuffle=True)
		cnt+=3

		if flag:
			train_list.append(dataset["train"])
			test_list.append(dataset["test"])

	if dataset_name.find("agnews")!=-1:
		os.chdir('./')
		dataset = datasets.load_from_disk("./agnews")
		idx2la = dataset.features["label"].names
		idx2la = {1:cnt,2:cnt+1,3:cnt+2,0:cnt+3}
		data_text = [d for d in dataset[:]["text"]]
		data_label = [idx2la[d] for d in dataset[:]["label"]]
		data = {"text": data_text,"label": data_label}
		dataset = Dataset.from_dict(data)
		dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=2002)
		cnt+=4

		if flag:
			train_list.append(dataset["train"])
			test_list.append(dataset["test"])
	
	if flag:
		sub_train_set = concatenate_datasets(train_list)
		sub_test_set = concatenate_datasets(test_list)
		dataset = datasets.DatasetDict({"train":sub_train_set,"test":sub_test_set})
		# print(dataset)
		# print(dataset["train"][0])

	if dataset_name.find("_fs")!=-1:
		ind = np.random.randint(len(dataset["train"]),size=32)
		sub_train_set = Dataset.from_dict(dataset['train'][ind])
		sub_test_set = dataset["test"]
		dataset = datasets.DatasetDict({"train":sub_train_set,"test":sub_test_set})
		# print(dataset)
		
		

	return dataset,cnt

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str, required=True, help='name of the dataset')
	args = parser.parse_args()
	dataset = args.dataset
	dataset = get_dataset(dataset,"<SEP>")