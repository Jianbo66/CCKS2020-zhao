import pickle
from NER_MODEL import *
from transformers import BertTokenizer, RobertaForTokenClassification
tokenizer = BertTokenizer.from_pretrained("RoBERTa_zh_Large_PyTorch")

max_seq_len = 40
train_corpus = pickle.load(open('./data/corpus_train.pkl', 'rb'))
train_questions = [train_corpus[i]['question'] for i in range(len(train_corpus))]
train_entitys = [train_corpus[i]['gold_entitys'] for i in range(len(train_corpus))]
train_entitys = [[entity[1:-1].split('_')[0] for entity in line] for line in train_entitys]

test_corpus = pickle.load(open('./data/corpus_test.pkl', 'rb'))
test_questions = [test_corpus[i]['question'] for i in range(len(test_corpus))]
test_entitys = [test_corpus[i]['gold_entitys'] for i in range(len(test_corpus))]
test_entitys = [[entity[1:-1].split('_')[0] for entity in line] for line in test_entitys]

def find_lcsubstr(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
            if m[i + 1][j + 1] > mmax:
                mmax = m[i + 1][j + 1]
                p = i + 1
    return s1[p - mmax:p]

def GetXY(questions, entitys):
    X1, X2,X3, Y = [], [], [], []
    for i in range(len(questions)):
        q = questions[i]
        #x1, x2 = tokenizer.encode(first=q, max_len=max_seq_len)  # 分别是 词索引序列和分块索引序列
        encoded_dict = tokenizer(q, max_length =max_seq_len,pad_to_max_length=True, return_tensors='pt')  # 分别是 词索引序列和分块索引序列
        x1, x2, x3 = encoded_dict["input_ids"][0],encoded_dict["token_type_ids"][0],encoded_dict["attention_mask"][0]
        y = [[0] for j in range(max_seq_len)]

        assert len(x1) == len(y)
        for e in entitys[i]:
            # 得到实体名和问题的最长连续公共子串
            e = find_lcsubstr(e, q)
            if e in q:
                begin = q.index(e) + 1
                end = begin + len(e)
                if end < max_seq_len - 1:
                    for pos in range(begin, end):
                        y[pos] = [1]

        X1.append(x1.tolist())
        X2.append(x2.tolist())
        X3.append(x3.tolist())
        Y.append(y)
    X1 = torch.tensor(X1).long()
    X2 = torch.tensor(X2).long()
    X3 = torch.tensor(X3).long()
    Y = torch.tensor(np.array(Y)).squeeze().long()

    return X1, X2, X3, Y

trainx1, trainx2, trainx3,trainy = GetXY(train_questions, train_entitys)  # (num_sample,max_len)
testx1, testx2,testx3, testy = GetXY(test_questions, test_entitys)

import torch
from torch.utils.data import TensorDataset, random_split

# 把input 放入 TensorDataset。
train_dataset = TensorDataset(trainx1, trainx2, trainx3,trainy)
test_dataset = TensorDataset(testx1, testx2, testx3,testy)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# 推荐batch_size 为 16 或者 32
batch_size = 8

# 为训练数据集和验证数据集设计DataLoaders.
train_dataloader = DataLoader(
            train_dataset,  # 训练数据.
            sampler = RandomSampler(train_dataset), # 打乱顺序
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            test_dataset, # 验证数据.
            sampler = RandomSampler(test_dataset), # 打乱顺序
            batch_size = batch_size
        )

model = enity_identifing(tokenizer.vocab_size,1000,bert_model)

# AdamW 是一个 huggingface library 的类，'W' 是'Weight Decay fix"的意思。
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - 默认是 5e-5
                  eps = 1e-8 # args.adam_epsilon  - 默认是 1e-8， 是为了防止衰减率分母除到0
                )

from transformers import get_linear_schedule_with_warmup

# bert 推荐 epochs 在2到4之间为好。
epochs = 50

# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels,attention):

    scores = (preds*attention == labels*attention)
    rights = 0
    for score in scores:
        if sum(score) == len(labels[0]):
            rights += 1
    #return np.sum((pred_flat == labels_flat)*atten)/ np.sum(atten)
    return rights/len(labels)

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

output_dir = './ner1/'
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
# 代码参考 https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# 设置随机种子.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 记录training ,validation loss ,validation accuracy and timings.
maxf = 0.0

def computeF(gold_entity, pre_entity):
    '''
    根据标注的实体位置和预测的实体位置，计算prf,完全匹配
    输入： Python-list  3D，值为每个实体的起始位置列表[begin，end]
    输出： float
    '''
    truenum = 0
    prenum = 0
    goldnum = 0
    for i in range(len(gold_entity)):
        goldnum += len(gold_entity[i])
        prenum += len(pre_entity[i])
        truenum += len(set(gold_entity[i]).intersection(set(pre_entity[i])))
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall / (precise + recall))
    except:
        precise = recall = f = 0.0
    print('本轮实体的F值是 %f' % (f))
    return precise, recall, f

def restore_entity_from_labels_on_corpus(predicty, questions):
    def restore_entity_from_labels(labels, question):
        entitys = []
        str = ''
        labels = labels[1:-1]
        for i in range(min(len(labels), len(question))):
            if labels[i] == 1:
                str += question[i]
            else:
                if len(str):
                    entitys.append(str)
                    str = ''
        if len(str):
            entitys.append(str)
        return entitys

    all_entitys = []
    for i in range(len(predicty)):
        all_entitys.append(restore_entity_from_labels(predicty[i], questions[i]))
    return all_entitys

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
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # `batch` 包括3个 tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_type = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # 清空梯度
            model.zero_grad()

            # forward
            # 参考 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            loss, logits = model(b_input_ids,b_input_type, b_input_mask, b_labels)

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

            predicty = np.array([[1. if each > 0.5 else 0 for each in line] for line in logit])
            # 计算training 句子的准确度.
            total_train_accuracy += flat_accuracy(predicty, label_id, attention_mask)

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

        for batch in validation_dataloader:
            # `batch` 包括3个 tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_type = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # 在valuation 状态，不更新权值，不改变计算图
            with torch.no_grad():
                # 参考 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                loss, logits = model(b_input_ids, b_input_type, b_input_mask, b_labels)

            # 计算 validation loss.
            total_eval_loss += loss.item()
            logit = logits.detach().cpu().numpy()
            label_id = b_labels.to('cpu').numpy()
            attention_mask = b_input_mask.cpu().numpy()
            predicty = np.array([[1 if each > 0.5 else 0 for each in line] for line in logit])
            # 计算 validation 句子的准确度.
            total_eval_accuracy += flat_accuracy(predicty, label_id, attention_mask)

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

# 训练模型
#train()

model.load_state_dict(torch.load('./ner1/pytorch_model.bin'))

for batch in validation_dataloader:
    # `batch` 包括3个 tensors:
    #   [0]: input ids
    #   [1]: attention masks
    #   [2]: labels
    b_input_ids = batch[0].to(device)
    b_input_type = batch[1].to(device)
    b_input_mask = batch[2].to(device)
    b_labels = batch[3].to(device)

    # 在valuation 状态，不更新权值，不改变计算图
    with torch.no_grad():
        # 参考 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        loss, logits = model(b_input_ids, b_input_type, b_input_mask, b_labels)
    logit = logits.detach().cpu().numpy()
    label_id = b_labels.to('cpu').numpy()
    attention_mask = b_input_mask.cpu().numpy()

    predicty = np.array([[1. if each > 0.5 else 0 for each in line] for line in logit])
    print(predicty)
    print(b_input_ids)

    predict_entitys = restore_entity_from_labels_on_corpus(predicty, b_input_ids)
    test_entitys = restore_entity_from_labels_on_corpus(label_id, b_input_ids)
    for j in range(0, 8):
        print(tokenizer.convert_ids_to_tokens(b_input_ids[j])[1:-1])
        print(predict_entitys[j])
        print(test_entitys[j])

    break