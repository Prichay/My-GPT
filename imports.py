from transformers import PreTrainedTokenizerFast
from tokenizers.models import WordPiece
from datasets import load_dataset
import os
from transformers import AutoModel, AutoTokenizer
import torch
import math