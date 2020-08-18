import torch
import torch.nn as nn
import numpy as np
from transformers import BertForTokenClassification, AdamW, BertConfig
from transformers import BertTokenizer, RobertaForTokenClassification,DataProcessor

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained("RoBERTa_zh_Large_PyTorch")

bert_model = RobertaForTokenClassification.from_pretrained(
    "./RoBERTa_zh_Large_PyTorch/",  # 使用 12-layer 的 BERT 模型.
    num_labels = 5000, # 多分类任务的输出标签为 len(tag2idx)个.
    output_attentions = False, # 不返回 attentions weights.
    output_hidden_states = False, # 不返回 all hidden-states.
)

class enity_identifing(nn.Module):
    def __init__(self,vocab_size, embedding_dim,bert_model):
        super(enity_identifing, self).__init__()
        self.bert_model = bert_model.to(device)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.rnn_type = "LSTM"
        self.nhid = 512
        self.rnn = nn.LSTM(5000, self.nhid, bidirectional=True, dropout=0.5).to(device)
        self.output = nn.Linear(2 * self.nhid, 1).to(device)
        self.loss_fn = nn.BCEWithLogitsLoss().to(device)
        self.sig = nn.Sigmoid().to(device)

       # self.sig = nn.LogSigmoid().to(device)

    def forward(self, inputs, type_ids,mask ,y):
        #  输入bert
        out = self.bert_model(inputs, type_ids, mask)
        #  输入LSTM
        hidden, states = self.rnn(out[0].contiguous())
        logits = self.output(hidden)

        loss = self.loss_fn(logits.squeeze(),y.float())*mask
        loss = (torch.sum(loss) / torch.sum(mask))
        logits = self.sig(logits.squeeze())
        return loss, logits

    def predicate(self, inputs, type_ids, mask ):
        out = self.bert_model(inputs, type_ids, mask)
        #  输入LSTM
        hidden, states = self.rnn(out[0].contiguous())
        logits = self.output(hidden)
        logits = self.sig(logits.squeeze())
        return logits

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
        question = tokenizer.convert_ids_to_tokens(question)
        entitys = []
        str = ''
        labels = labels[1:-1]
        question = question[1:-1]
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