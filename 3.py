#coding=utf-8
import jieba
import chardet
import re
import pymysql
import csv
from gensim.models import word2vec
import pandas as pd

#清洗数据函数
#通过正则表达式来简单清洗数据
def clean(text):
    #去除正文中@、回复、//转发中的用户名
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)
    #去除话题内容
    # text = re.sub(r"#\S+#", "", text)
    #去除转发微博这种词
    text = text.replace("转发微博", "")
    #去除网址
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text) 
    # 去除表情符号
    text = re.sub(r"\[\S+\]", "", text)
    # 合并正文中过多的空格
    text = re.sub(r"\s+", " ", text) 
    # strip方法，用于移除字符串头尾指定的字符，默认的是空格或者换行符。只能删除头尾不能删中间。
    return text.strip()

#stopwordsList函数
#导入停用词表，放在stopwords中
def stopWordsList(filepath):
    stopwords=[]
    # stopwords = [line.restrip()for line in open(filepath,'r',encoding='utf-8').readlines]
    with open(filepath,'r',encoding='utf-8')as K:
        for line in K:
            line = line.strip()
            stopwords.append(line)
    return stopwords


stopwords = stopWordsList('hit_stopwords.txt')
with open('test.csv','r',encoding='utf-8') as f:
    for line in f:
        # 切割字符串，提出最后一个逗号后的内容，那个是用户名
        # print(line.split(',')[-1])  
        # 将除了用户名的东西提出来，然后join成一个字符串
        line = clean(line)
        sentence = line.split(',')[:-1]
        result = ''.join(sentence)
        
        # with open('testClean.txt','a',encoding='utf-8') as C:
        #     C.write(result+'\n')
        with open('testJieba2.txt','a',encoding='utf-8') as k:
            #分词
            word = (" ".join(jieba.lcut(result)))
            final = " "
            #停用词过滤
            for wordone in word:
                if wordone not in stopwords:
                    final+= wordone
            k.write(final)
            #以下是之前jieba的三个模式测试
            # k.write("/".join(jieba.lcut(result))+'\n')  #精确模式好像是
            # k.write("/".join(jieba.lcut(result,cut_all=True))+'\n')  #全模式
            # k.write("/".join(jieba.lcut_for_search(result))+'\n')  #搜索引擎模式
            k.write('\n')



# #下面是用数据集里的尝试清洗的
# # 连接数据库
# DBHOST = 'localhost'
# DBUSER = 'root'
# DBPASS = 'lulu123'
# DBNAME = 'test'
# try:
#     db = pymysql.connect(DBHOST,DBUSER,DBPASS,DBNAME)
#     print("数据库连接成功！")
# except pymysql.Error as e:
#     print("数据库连接失败："+str(e))

# # 声明一个游标
# cur = db.cursor()

# # 查询
# sqlQuery = "SELECT text FROM weibo limit 0,100"
# try:
#     cur.execute(sqlQuery)
#     results = cur.fetchall()
#     for result in results:

#         result=str(result)
#         with open('testForBig.txt','a',encoding='utf-8') as C:
#             C.write(result+'\n')

#         line = clean(str(result))
#         with open('testForBigClean.txt','a',encoding='utf-8') as k:
#             k.write(line+'\n')
# except pymysql.Error as e:
#     print("数据库查询失败"+str(e))


#word2vec使用
#加载语料
data = word2vec.Text8Corpus("testJieba2.txt")
#训练语料
path  = open("model.txt",'w',encoding='utf-8')
#一些参数注释：
#data：数据源
#hs：=1时：采用基于hierarchical softmax的技巧；=0时：用的negative sampling
# min_count:对字典做截断，词频少于mincout的会被舍弃
# window:表示当前此语预测词在一个句子中的最大距离是多少
# size：特征向量的维度，默认100，size越大需要的训练数据越大，效果越好
model = word2vec.Word2Vec(data, hs=1, min_count=1, window=10, size=100)
#保存模型至model.model
model.save('model.model')
#预测
try:
    sim2 = model.similarity(u'广东',u'电报')
except KeyError:
    sim2=0
    print('没有喔')
#打出sim2：两者的相似度
print(u'牛奶和面膜的相似度为：',sim2)
print("----------------------")
#读取模型，下次就不用再训练了
model = word2vec.Word2Vec.load('model.model')
print(model['电报'])