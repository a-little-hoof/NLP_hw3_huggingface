import sys
import logging
import random
from dataHelper import get_dataset
from dataclasses import dataclass, field
import wandb

import datasets
import evaluate
import numpy as np

import transformers
from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	EvalPrediction,
	HfArgumentParser,
	Trainer,
	TrainingArguments,
	default_data_collator,
	set_seed,
)

from config import DataTrainingArguments,ModelArguments

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

import adapters

logger = logging.getLogger(__name__)

def main():
	'''
		initialize logging, seed, argparse...
	'''
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	print(parser.parse_args_into_dataclasses())
	
	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	if training_args.should_log:
		# The default of training_args.log_level is passive, so we set log level at info here to have that default.
		transformers.utils.logging.set_verbosity_info()

	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
		+ f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
	)
	logger.info(f"Training/evaluation parameters {training_args}")

	# Set seed before initializing model.
	set_seed(training_args.seed)
	name = str(training_args.seed)+"-"+model_args.model_name_or_path+"-"+data_args.dataset_name
	wandb.init(name=name+"-adapter" if model_args.use_adapter else name+"-lora" if model_args.use_lora else name)

	'''
		load datasets
	'''

	raw_datasets, label_num = get_dataset(data_args.dataset_name,"<SEP>")
			
	'''
		load models
	'''
	config = AutoConfig.from_pretrained(
		model_args.model_name_or_path, # 'bert-base-cased'
		num_labels=label_num,
		cache_dir=model_args.cache_dir, # None
	)
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast=True,
	)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
	)
	
	if model_args.use_lora:
		peft_config = LoraConfig(
			task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
		)
		model = get_peft_model(model, peft_config)
		model.print_trainable_parameters()
		
	if model_args.use_adapter:
		adapters.init(model)
		model.add_adapter("bottleneck_adapter")
		model.train_adapter(["bottleneck_adapter"])

	trainable_model_params = model.num_parameters(only_trainable=True)
	print("parameters",trainable_model_params)
	wandb.config.update({"Model Parameters": trainable_model_params})
	'''
		process datasets and build up datacollator
	'''
	max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
	print("model_max_length:",tokenizer.model_max_length)
	def preprocess_function(examples):
		# Tokenize the texts
		args = (
			(examples["text"],)
		)
		result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)
		return result


	with training_args.main_process_first(desc="dataset map pre-processing"):
		raw_datasets = raw_datasets.map(
			preprocess_function,
			batched=True,
			load_from_cache_file=True,
			desc="Running tokenizer on dataset",
		)
	
	train_dataset = raw_datasets["train"]
	test_dataset = raw_datasets["test"]

	# Log a few random samples from the training set:
	for index in random.sample(range(len(train_dataset)), 3):
		logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

	# Get the metric function
	accuracy = evaluate.load("./accuracy.py")
	micro = evaluate.load("./metric_microf1.py")
	macro = evaluate.load("./metric_macrof1.py")

	# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
	# predictions and label_ids field) and has to return a dictionary string to float.
	def compute_metrics(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		preds = np.argmax(preds, axis=1)
		result = dict()
		result.update(accuracy.compute(predictions=preds, references=p.label_ids))
		result.update(micro.compute(predictions=preds, references=p.label_ids))
		result.update(macro.compute(predictions=preds, references=p.label_ids))
		return result


	'''
		Training!
	'''
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset, 
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
		data_collator=default_data_collator,
	)

	# Training
	if training_args.do_train:
		checkpoint = None
		if training_args.resume_from_checkpoint is not None:
			checkpoint = training_args.resume_from_checkpoint
		train_result = trainer.train(resume_from_checkpoint=checkpoint)
		metrics = train_result.metrics
		max_train_samples = (
			len(train_dataset)
		)
		metrics["train_samples"] = min(max_train_samples, len(train_dataset))

		trainer.save_model()  # Saves the tokenizer too for easy upload
		if model_args.use_adapter:
			model.save_adapter(training_args.output_dir, "bottleneck_adapter")

		trainer.log_metrics("train", metrics)
		trainer.save_metrics("train", metrics)
		trainer.save_state()

	# Evaluation
	if training_args.do_eval:
		logger.info("*** Evaluate ***")

		# Loop to handle MNLI double evaluation (matched, mis-matched)
		eval_datasets = [test_dataset]

		for eval_dataset in eval_datasets:
			metrics = trainer.evaluate(eval_dataset=eval_dataset)

			max_eval_samples = (
				len(eval_dataset)
			)
			metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

			trainer.log_metrics("eval", metrics)
			trainer.save_metrics("eval",metrics)


if __name__ == "__main__":
	main()