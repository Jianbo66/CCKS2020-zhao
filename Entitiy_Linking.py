import  re
import pandas as pd
import pickle
from NER_MODEL import *
#from simlarity_model import  *
from transformers import (
    AdamW,
    BertTokenizer,
    BertForNextSentencePrediction
)
INPUT_PATH = './data/PKUBASE/pkubase-complete-2020/pkubase-complete.txt'
#INPUT_PATH = './data/pkubase-complete2.txt'
PICKLE_PATH = './data/pickle/'

sim_model = BertForNextSentencePrediction.from_pretrained("bert-base-chinese")
sim_model = sim_model
sim_model.load_state_dict(torch.load('./model/simlarity1/pytorch_model.bin'))

class Entities_processor(DataProcessor):
    """生成全局变量Pickle文件的预处理器"""

    def __init__(self):
        self.entities = dict()
        self.type = set()

    def run(self,INPUT_PATH):

        with open(INPUT_PATH, encoding="utf-8") as f:
            try:
               while True:
                   line = f.readline()
                   entity = line.split("\t")[0][1:-1]

                   if line and entity is not "":
                       entiy_tuple = line.split("\t")[0][1:-1] + "-" + line.split("\t")[1][1:-1]
                       self.entities[entiy_tuple] = (line.split("\t")[0][1:-1] , line.split("\t")[1][1:-1], line.split("\t")[2][1:-4])

                   else:
                       break
            finally:
               f.close()
            # 保存pickle文件
            pd.to_pickle(self.entities, PICKLE_PATH + 'ENTITI_ANSER.pkl')

    def Likely_Enities(self, INPUT_PATH):

        self.qestions = self._read_tsv(INPUT_PATH)[1:]
        self.predict_entitys = []
        self.qestions = [self.qestions[i][0] for i in range(0,len(self.qestions))]
        self.inputs = tokenizer(self.qestions,pad_to_max_length=True, return_tensors='pt')
        self.model = enity_identifing(tokenizer.vocab_size,1000,bert_model)
        self.model.load_state_dict(torch.load('./ner1/pytorch_model.bin'))

        for i in range(0,self.inputs["input_ids"].size(0)):
            with torch.no_grad():
                logits = self.model.predicate(self.inputs["input_ids"][i].unsqueeze(0).to(device),
                                              self.inputs["token_type_ids"][i].unsqueeze(0).to(device),
                                              self.inputs["attention_mask"][i].unsqueeze(0).to(device))

            logit = logits.unsqueeze(0).detach().cpu().numpy()
            predicty = np.array([[1. if each > 0.5 else 0 for each in line] for line in logit])

            predict_entity = restore_entity_from_labels_on_corpus(predicty, self.inputs["input_ids"][i].unsqueeze(0))
            print(self.qestions[i])
            print(predict_entity)
            self.predict_entitys.append(predict_entity)

        return self.qestions, self.predict_entitys

    def Identify_Entities(self,qestion,predict_entity):

        Entities_Answers = pickle.load(open('./data/pickle/ENTITI_ANSER.pkl', 'rb'))
        All_Entities = list(Entities_Answers.keys())
        Enities = list()
        print(qestion)
        high_score = 0
        high_score_entity = predict_entity[0][0]
        count_answer = dict()
        for i in range(0,len(predict_entity[0])):
            print(i)
            print(predict_entity[0][i])

            print("start")
            if len(predict_entity[0][i]) == 1:
                continue

            for entity in All_Entities:

                if entity.find(predict_entity[0][i]) != -1:
                    encoding = tokenizer(qestion, Entities_Answers[entity][0]+"|||"+Entities_Answers[entity][1], return_tensors='pt')
                    logits = sim_model(**encoding)
                    if torch.argmax(logits[0],dim=-1) == 0:
                        # 0 代表分数最有可能的答案，记录分数最高的答案
                        if Entities_Answers[entity][2] not in count_answer:
                            count_answer[Entities_Answers[entity][2]] = 1
                        else:
                            count_answer[Entities_Answers[entity][2]] += 1

                        if logits[0][0][0] -logits[0][0][1] > high_score:
                            high_score = logits[0][0][0]-logits[0][0][1]
                            high_score_entity = entity
                        #if qestion.find(entity) != -1:
                        print(logits)
                        print(entity)
                        print(Entities_Answers[entity])
                        print("-------")
        print("AAAAA")
        print(high_score)
        print(high_score_entity)
        print(Entities_Answers[high_score_entity])
        sort_tuple_list = sorted([(value, key) for (key, value) in count_answer.items()])
        print("eeee")
                #pass
                #print(entity)
                # if entity.find(predict_entity[0][i]) != -1:
                #     #print("54645")
                #     # if qestion.find(entity) != -1:
                #     Enities.append(entity)
                #     print(entity)
                #     print(Entities_Answers[entity])
        #print(set(Enities))

        Enities = list(set(Enities))


        return Enities

    def Entities_Filter(self,Enities):

        pass

processor = Entities_processor()
#processor.run(INPUT_PATH)
# print("---")

#processor.Likly_Enities("./data/csv/task1-4_valid_2020.questions.csv")
# qestion = "风湿热、肾炎、一般发热属于什么？"
# predict_entity = [['风湿热', '肾炎', '般', '热']]




qestion = "香奈儿五号香水是什么时候发行的？"
predict_entity = [['香奈儿','发行']]
processor.Identify_Entities(qestion,predict_entity)
print("---")