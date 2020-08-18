from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from enum import Enum
import os
import numpy as np

#from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, PreTrainedTokenizer
from transformers import (
    AdamW,
    BertTokenizer,
    BertForNextSentencePrediction
)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

model = BertForNextSentencePrediction.from_pretrained("bert-base-chinese")
model = model.cuda()