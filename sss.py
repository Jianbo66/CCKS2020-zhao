# import codecs as cs
# import re
# import time
# import pickle
# import math
import pandas as pd


INPUT_PATH = './data/PKUBASE/pkubase-complete-2020/pkubase-complete.txt'
PICKLE_PATH = './data/pickle/'
# <莫妮卡·贝鲁奇> <代表作品> <主要作品> <湖上草>  <叶文洁> <毕业院校>
class Entities_processor:
    """生成全局变量Pickle文件的预处理器"""

    def __init__(self):
        self.entities = set()
        self.type = set()

    def run(self,INPUT_PATH):

        with open(INPUT_PATH, encoding="utf-8") as f:
            try:
               while True:
                   line = f.readline()
                   if line and line.split("\t")[0][1:-1] is not "":
                       self.entities.add(line.split("\t")[0][1:-1])
                       self.type.add(line.split("\t")[1])
                   else:
                      break
            finally:
               f.close()
            # 保存pickle文件
            pd.to_pickle(self.entities, PICKLE_PATH + 'ENTITIS.pkl')
            pd.to_pickle(self.type, PICKLE_PATH + 'TYPES.pkl')

processor = Entities_processor()
processor.run(INPUT_PATH)

print("---")