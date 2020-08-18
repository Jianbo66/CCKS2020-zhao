import codecs as cs
import re
# import time
import pickle
import math




inputpath = './data/PKUBASE/pkubase-mention2ent-2020/pkubase-mention2ent.txt'
# <莫妮卡·贝鲁奇> <代表作品> <主要作品> <湖上草>  <叶文洁> <毕业院校>
with open(inputpath, encoding="utf-8") as f:
    try:
       while True:
           line = f.readline()
           if line:
              #print(type(line))
              #if line.split("\t")[0]== "<莫妮卡·贝鲁奇>" and line.split("\t")[1]== "<代表作品>":
              if line.split("\t")[0] == "清华大学" :
                    print("line= ",line)
           else:
              break
    finally:
       f.close()


    print("---")