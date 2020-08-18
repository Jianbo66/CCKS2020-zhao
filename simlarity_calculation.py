from simlarity_model import  *
""" 创建一个加载THUCNews数据集的库
"""
class NextSentenceProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sentence_simlarity_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sentence_simlarity_test.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sentence_simlarity_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        #return list(range(1, 15))
        return [0,1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        lines = lines[1:]
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0][1:]
            text_b = line[1]
            #label = None if test_mode else int(line[0])
            label = line[2][:-1]
            # 这里的InputExample是一个非常简单的类，仅仅包含了text_a, text_b和label三个部分
            # https://github.com/huggingface/transformers/blob/master/src/transformers/data/processors/utils.py#L31
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

processor = NextSentenceProcessor()
PATH = "./data/csv/"
Train_examples = processor.get_train_examples(PATH)
Test_examples = processor.get_test_examples(PATH)

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

""" 在这个function中，我们会把文本数据转化为可以传入BERT模型的index, mask等输入
"""
def convert_examples_to_features( examples,tokenizer,max_length= None,label_list=None,output_mode=None):
    if max_length is None:
        max_length = tokenizer.max_len

    processor = NextSentenceProcessor()
    if label_list is None:
        label_list = processor.get_labels()

    if output_mode is None:
        output_mode = "classification"

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[int(example.label)]
        elif output_mode == "regression":
            return float(int(example.label))
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        # https://github.com/huggingface/transformers/blob/master/src/transformers/data/processors/utils.py#L56
        # InputFeatures当中包含了input_ids, attention_mask, token_type_ids和label四个部分
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features

Train_features = convert_examples_to_features(Train_examples,tokenizer,150)
Test_features = convert_examples_to_features(Test_examples,tokenizer,150)

import torch
import numpy
from torch.utils.data import TensorDataset, random_split
def build_dataset(features):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    train_y = []
    for feature in features:
        input_ids.append(feature.input_ids)
        attention_mask.append(feature.attention_mask)
        token_type_ids.append(feature.token_type_ids)
        train_y.append(feature.label)

    input_ids = torch.from_numpy(numpy.array(input_ids)).long()
    attention_mask = torch.from_numpy(numpy.array(attention_mask)).long()
    token_type_ids = torch.from_numpy(numpy.array(token_type_ids)).long()
    train_y = torch.from_numpy(numpy.array(train_y)).long()
    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, train_y)
    return dataset

train_set = build_dataset(Train_features)
test_set = build_dataset(Test_features)


from torch.utils.data import TensorDataset, DataLoader

train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
validation_dataloader = DataLoader(test_set, batch_size=8, shuffle=True)

# print("--------")

# AdamW 是一个 huggingface library 的类，'W' 是'Weight Decay fix"的意思。
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - 默认是 5e-5
                  eps = 1e-8 # args.adam_epsilon  - 默认是 1e-8， 是为了防止衰减率分母除到0
                )

from transformers import get_linear_schedule_with_warmup

# bert 推荐 epochs 在2到4之间为好。
epochs = 5

# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


def flat_accuracy(preds, labels,attention):

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time

import datetime
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # 返回 hh:mm:ss 形式的时间
    return str(datetime.timedelta(seconds=elapsed_rounded))

import os
import random
import numpy as np
from transformers import WEIGHTS_NAME, CONFIG_NAME

output_dir = './model/simlarity1/'
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
# 代码参考 https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# 设置随机种子.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 记录training ,validation loss ,validation accuracy and timings.

def train():
    training_stats = []
    # 设置总时间.
    total_t0 = time.time()
    best_val_accuracy = 0

    for epoch_i in range(0, epochs):
        print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

        # 记录每个 epoch 所用的时间
        t0 = time.time()
        total_train_loss = 0
        total_train_accuracy = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            # 每隔40个batch 输出一下所用时间.
            if step % 300 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # `batch` 包括3个 tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].cuda()
            b_input_type = batch[1].cuda()
            b_input_mask = batch[2].cuda()
            b_labels = batch[3].cuda()

            # 清空梯度
            model.zero_grad()

            # forward
            # 参考 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            loss, logits = model(b_input_ids,b_input_type, b_input_mask, next_sentence_label = b_labels)

            total_train_loss += loss.item()

            # backward 更新 gradients.
            loss.backward()

            # 减去大于1 的梯度，将其设为 1.0, 以防梯度爆炸.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新模型参数
            optimizer.step()

            # 更新 learning rate.
            scheduler.step()

            logit = logits.detach().cpu().numpy()
            label_id = b_labels.to('cpu').numpy()
            attention_mask = b_input_mask.cpu().numpy()

            #predicty = np.array([[1. if each > 0.5 else 0 for each in line] for line in logit])
            # 计算training 句子的准确度.
            total_train_accuracy += flat_accuracy(logit, label_id, attention_mask)


        # 计算batches的平均损失.
        avg_train_loss = total_train_loss / len(train_dataloader)
        # 计算训练时间.
        training_time = format_time(time.time() - t0)

        # 训练集的准确率.
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print("  训练准确率: {0:.2f}".format(avg_train_accuracy))
        print("  平均训练损失 loss: {0:.2f}".format(avg_train_loss))
        print("  训练时间: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        t0 = time.time()

        # 设置 model 为valuation 状态，在valuation状态 dropout layers 的dropout rate会不同
        model.eval()

        # 设置参数
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            # `batch` 包括3个 tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].cuda()
            b_input_type = batch[1].cuda()
            b_input_mask = batch[2].cuda()
            b_labels = batch[3].cuda()

            # 在valuation 状态，不更新权值，不改变计算图
            with torch.no_grad():
                # 参考 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                #loss, logits = model(b_input_ids, b_input_type, b_input_mask, b_labels)
                loss, logits = model(b_input_ids, b_input_type, b_input_mask, next_sentence_label=b_labels)
            # 计算 validation loss.
            total_eval_loss += loss.item()
            logit = logits.detach().cpu().numpy()
            label_id = b_labels.to('cpu').numpy()
            attention_mask = b_input_mask.cpu().numpy()
            #predicty = np.array([[1 if each > 0.5 else 0 for each in line] for line in logit])
            # 计算 validation 句子的准确度.
            total_eval_accuracy += flat_accuracy(logit, label_id, attention_mask)

        # 计算 validation 的准确率.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("")
        print("  测试准确率: {0:.2f}".format(avg_val_accuracy))

        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), output_model_file)
            #model.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir)

        # 计算batches的平均损失.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # 计算validation 时间.
        validation_time = format_time(time.time() - t0)

        print("  平均测试损失 Loss: {0:.2f}".format(avg_val_loss))
        print("  测试时间: {:}".format(validation_time))

        # 记录模型参数
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("训练一共用了 {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
#
# 训练模型

train()
print("---")

