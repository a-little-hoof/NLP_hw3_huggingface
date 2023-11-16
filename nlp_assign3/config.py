from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.

	Using `HfArgumentParser` we can turn this class
	into argparse arguments to be able to specify them on
	the command line.
	"""
	dataset_name: str = field(
		default=None,
		metadata={
			"help": (
				"Name of the dataset you choose."
			)
		},
	)
	max_seq_length: int = field(
		default=128,
		metadata={
			"help": (
				"The maximum total input sequence length after tokenization. Sequences longer "
				"than this will be truncated, sequences shorter will be padded."
			)
		},
	)


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	use_adapter: bool = field(
		default=False,
		metadata={"help": "Whether to use the bottleneck adapter."},
	)
	use_lora: bool = field(
		default=False,
		metadata={"help": "Whether to use LoRA."},
	)
