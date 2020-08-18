# -*- coding: utf-8 -*-
"""
Created on Sun Aug 2020

@author: FRANK
"""
import codecs as cs
import re
import pickle
import math
import pandas as pd
import numpy as np
PICKLE_PATH = './data/pickle/'
ENTITIE_TO_TAGS = []

def LoadCorpus(path):

    def writefile(text):

        corpus = {}
        question_num = 0
        e1hop1_num = 0
        e1hop2_num = 0
        e2hop2_num = 0
        nu = 0
        for i in range(len(text)):
            # 对问题进行预处理
            question = text[i].split('\r\n')[0].split(':')[1]
            question = re.sub('我想知道', '', question)
            question = re.sub('你了解', '', question)
            question = re.sub('请问', '', question)

            answer = text[i].split('\n')[2].split('\t')
            sql = text[i].split('\n')[1]
            sql = re.findall('{.+}', sql)[0]
            elements = re.findall('<.+?>|\".+?\"|\?\D', sql) + re.findall('\".+?\"', sql)
            # elements中包含创引号的项目可能有重复，需要去重
            new_elements = []
            for e in elements:
                if e[0] == '\"':
                    if e not in new_elements:
                        new_elements.append(e)
                else:
                    new_elements.append(e)
            elements = new_elements
            gold_entitys = []
            gold_relations = []
            for j in range(len(elements)):
                if elements[j][0] == '<' or elements[j][0] == '\"':
                    if j % 3 == 1:
                        gold_relations.append(elements[j])
                    else:
                        gold_entitys.append(elements[j])

            gold_tuple = tuple(gold_entitys + gold_relations)
            dic = {}
            dic['question'] = question  # 问题字符串
            dic['answer'] = answer  # 问题的答案
            dic['gold_tuple'] = gold_tuple
            dic['gold_entitys'] = gold_entitys
            dic['gold_relations'] = gold_relations
            dic['sql'] = sql
            corpus[i] = dic
            Entity = gold_entitys[0][1:-1].split("_")[0]

            start_position = question.find(Entity)
            if start_position == -1:
                 Entity = Entity.lower().replace("·","")
                 start_position = question.find(Entity)
            if start_position is not -1: # -1 表示无对应实体

                end_position = start_position + len(Entity)
                # question_lstm = list(jieba.lcut(question))
                question_lstm = [question[i] for i in range(0,len(question))]
                question_tags = ["O"  for i in range(0,len(question))]
                question_tags[start_position] = "B"
                for i in range(start_position+1,end_position):
                    question_tags[i] = "I"

                ENTITIE_TO_TAGS.append((question_lstm,question_tags))

            if start_position == -1:
                nu += 1
            # 一些统计信息
            if len(gold_entitys) == 1 and len(gold_relations) == 1:
                e1hop1_num += 1
            elif len(gold_entitys) == 1 and len(gold_relations) == 2:
                e1hop2_num += 1
            elif len(gold_entitys) == 2 and len(gold_relations) == 2:
                e2hop2_num += 1
            elif len(gold_entitys) == 2 and len(gold_relations) < 2:
                print(elements)
                print(dic['gold_entitys'])
                print(dic['sql'])
                print('\n')
            question_num += 1

        print('语料集问题数为%d==单实体单关系数为%d====单实体双关系数为%d==双实体双关系数为%d==总比例为%.3f\n' \
              % (question_num, e1hop1_num, e1hop2_num, e2hop2_num, (e1hop1_num + e1hop2_num + e2hop2_num) / question_num))
        print("无实体：", nu)
        return corpus

    with cs.open(path, 'r', 'utf-8') as fp:
        train_text = fp.read().split('\r\n\r\n')[:-1]
        length = len(train_text)

        #分出训练集和测试集
        train_corpus_length = math.ceil(0.2 * length)
        test_corpus_length = length - train_corpus_length
        train_corpus = train_text[0:train_corpus_length]
        test_corpus = train_text[test_corpus_length:]

        corpus = writefile(train_corpus)
        pickle.dump(corpus, open('./data/pickle/corpus_train.pkl', 'wb'))

        corpus = writefile(test_corpus)
        pickle.dump(corpus, open('./data/pickle/corpus_test.pkl', 'wb'))
        fp.close()

# INPUT_PATH = './data/task1-4_train_2020 (1).txt'
# LoadCorpus(INPUT_PATH)
# pickle.dump(ENTITIE_TO_TAGS, open('./data/pickle/ENTITIE_TO_TAGS.pkl', 'wb'))

# 问题和答案做打分bert 二分类的训练数据
def simlarity_sentences_examples():
    def load_data(PATH):

        train_corpus = pickle.load(open(PATH, 'rb'))
        train_questions = [train_corpus[i]['question'] for i in range(len(train_corpus))]
        train_entitys = [train_corpus[i]['gold_entitys'] for i in range(len(train_corpus))]
        train_entitys = [[entity[1:-1] for entity in line] for line in train_entitys]
        train_tuple = [train_corpus[i]['gold_tuple'] for i in range(len(train_corpus))]
        train_answer = [train_corpus[i]['answer'] for i in range(len(train_corpus))]
      #  train_sql = [train_corpus[i]['sql'] for i in range(len(train_corpus))]
        return train_questions,train_entitys,train_tuple,train_answer

    Entities_Answers = pickle.load(open('./data/pickle/ENTITI_ANSER.pkl', 'rb'))
    Entities_Answers_List = [i for i in Entities_Answers.values()]

    #  训练集
    PATH = './data/corpus_train.pkl'
    questions,entitys,tuple, train_answer = load_data(PATH)
    sentences = list()


    for i in range(0,len(questions)):
        size = 10
        sentence = questions[i] + "\t" + tuple[i][0][1:-1] + "|||"+tuple[i][1][1:-1] + "|||"+train_answer[i][0][1:-1] + "\t" + "0"
        sentences.append(sentence)
        nu = 0
        for p in range(0,len(Entities_Answers_List)):
            if Entities_Answers_List[p][0].find(entitys[i][0]) != -1:
                #simlar_group.append(Entities_Answers_List[p])
                if Entities_Answers_List[p][2] != train_answer[i][0][1:-1]:
                    answer = Entities_Answers_List[p]
                    sentence = questions[i] + "\t" + answer[0] + "|||" + answer[1] + "|||" + answer[2] + "\t" + "1"
                    sentences.append(sentence)
                    print(sentence)
                    nu += 1
                    if nu > 20:
                        break

        neg = np.random.randint(len(Entities_Answers_List), size=size)

        for k in range(0,size):
            n = neg[k]
            answer = Entities_Answers_List[n]
            sentence = questions[i]+"\t"+answer[0] + "|||"+ answer[1]+ "|||" + answer[2] + "\t" + "1"
            sentences.append(sentence)

    link_data = pd.DataFrame(sentences)
    link_data.to_csv("./data/csv/sentence_simlarity_train.csv", index=False, sep='\t')

    # 测试集
    PATH = './data/corpus_test.pkl'
    questions, entitys, tuple, train_answer = load_data(PATH)
    sentences = list()
    for i in range(0, len(questions)):
        size = 10
        neg = np.random.randint(len(Entities_Answers_List), size=size)

        sentence = questions[i] + "\t" + tuple[i][0][1:-1] + "|||" + tuple[i][1][1:-1] + "|||" + train_answer[i][0][1:-1] + "\t" + "0"
        sentences.append(sentence)
        nu = 0
        for p in range(0,len(Entities_Answers_List)):
            if Entities_Answers_List[p][0].find(entitys[i][0]) != -1:
                if Entities_Answers_List[p][2] != train_answer[i][0][1:-1]:
                    answer = Entities_Answers_List[p]
                    sentence = questions[i] + "\t" + answer[0] + "|||" + answer[1] + "|||" + answer[2] + "\t" + "1"
                    sentences.append(sentence)
                    print(sentence)
                    nu += 1
                    if nu > 20:
                        break

        for k in range(0, size):
            n = neg[k]
            answer = Entities_Answers_List[n]
            sentence = questions[i] + "\t" + answer[0] + "|||" + answer[1] + "|||" + answer[2] + "\t" + "1"
            sentences.append(sentence)

    link_data = pd.DataFrame(sentences)
    link_data.to_csv("./data/csv/sentence_simlarity_test.csv", index=False, sep='\t')

simlarity_sentences_examples()



QUESTIONS_PATH = './data/task1-4_valid_2020.questions'
def TEST_QUESTIONS(QUESTIONS_PATH):
    questions = []
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        try:
            while True:
                line = f.readline()

                if line:
                    question = line.split(":")[1][:-1]
                    questions.append(question)
                else:
                    break
        finally:
            f.close()

    # 保存csv文件
    link_data = pd.DataFrame(questions)
    link_data.to_csv("./data/csv/task1-4_valid_2020.questions.csv", index=False, sep='\t')

#TEST_QUESTIONS(QUESTIONS_PATH)