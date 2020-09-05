import pandas as pd 
import jieba
import numpy as np 

news = pd.read_csv('sqlResult.csv',encoding='gb18030')

print(news.shape)
print(news.head())

news = news.dropna(subset=['content'])

print(news.shape)
#加载停用词
with open('chinese_stopwords.txt','r',encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]

    #分词
def split_text(text):
    text = text.replace(' ','').replace('\n','')
    text2 = jieba.cut(text)
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result
print(news.iloc[0].content)

print(split_text(news.iloc[0].content))

corpus = list(map(split_text,[str(i) for i in news.content]))

print(corpus)
#计算corpus的IFIDF

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
countVectorizer = CountVectorizer(encoding='gb18030',min_df=0.015)
TfidfTransformer = TfidfTransformer()
countvector = countVectorizer.fit_transform(corpus)

tfidf = TfidfTransformer.fit_transform(countvector)

print(tfidf.shape)

#标记是否为新华社新闻
label = list(map(lambda source: 1 if '新华社' in str(source) else 0,news.source))

print(label)
from sklearn.model_selection import train_test_split
#数据集切分
X_train,X_test,y_train,y_test = train_test_split(tfidf.toarray(),label,test_size=0.3)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,precision_score,recall_score
y_predict = model.predict(X_test)
print('准确率：',accuracy_score(y_test,y_predict))
print('精确率：',precision_score(y_test,y_predict))
print('召回率：',recall_score(y_test,y_predict))

#使用模型进行风格预测
prediction = model.predict(tfidf.toarray())
labels = np.array(label)

#compare_news_index有两列，prediction为预测风格，labels 为真实风格
compare_news_index = pd.DataFrame({'prediction':prediction,'labels':labels})
copy_news_index = compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['labels']==0)].index



#实际为新华社的新闻
xinhuashe_news_index = compare_news_index[(compare_news_index['labels']==1)].index
print('可能为copy的新闻条数：', len(copy_news_index))

from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
normalizer = Normalizer()
scaled_array = normalizer.fit_transform(tfidf.toarray())

#使用Kmeans进行 全量文档进行聚类
kmeans = KMeans(n_clusters=25)
k_labels = kmeans.fit_predict(scaled_array)
print(k_labels.shape)

#创建id_class,ID是1-87000，class是1-25
from collections import defaultdict
id_class = {index:class_ for index, class_ in enumerate(k_labels)}
class_id = defaultdict(set)
for index,class_ in id_class.items():
    #只统计新华社发布的class_id
    if index in xinhuashe_news_index.tolist():
        class_id[class_].add(index)

from sklearn.metrics.pairwise import cosine_similarity
#查找相似的文章
def find_similar_text(cpindex, top=10):
    #只在新华社发布的文章中查找
    dist_dict = {i:cosine_similarity(tfidf[cpindex],tfidf[i]) for i in class_id[id_class[cpindex]]}
    #从大小进行排序
    return sorted(dist_dict.items(),key=lambda x:x[1][0],reverse=True)[:top]
cpindex=3352
print('是否在新华社',cpindex in xinhuashe_news_index )
print('是否在copy_news', cpindex in copy_news_index)
similar_list = find_similar_text(cpindex)
print(similar_list)

print('怀疑抄袭：\n',news.iloc[cpindex].content)
#找一篇相似的原文
similar2 = similar_list[0][0]
print('相似的原文：\n',news.iloc[similar2].content)

#以下为结果部分

准确率： 0.880690737833595
精确率： 0.960230340111571
召回率： 0.9053274516457415
可能为copy的新闻条数： 2799
(87054,)
是否在新华社 False
是否在copy_news True
[(3134, array([[0.96849134]])), (63511, array([[0.94643198]])), (29441, array([[0.94283416]])), (3218, array([[0.87621892]])), (29615, array([[0.86936328]])), (29888, array([[0.86215862]])), (64046, array([[0.85278235]])), (29777, array([[0.84875422]])), (64758, array([[0.73394798]])), (29973, array([[0.7252432]]))]
怀疑抄袭：
 　　中国5月份56座城市新建商品住宅价格环比上涨，4月份为58座上涨。5月份15个一线和热点二线城市房地产市场基本稳定，5月份房地产调控政策效果继续显现。
　　统计局：15个一线和热点二线城市房价同比涨幅全部回落
　　国家统计局城市司高级统计师刘建伟解读5月份房价数据
　　5月份一二线城市房价平均涨幅继续回落
　　国家统计局今日发布了2017年5月份70个大中城市住宅销售价格统计数据。对此，国家统计局城市司高级统计师刘建伟进行了解读。
　　一、15个一线和热点二线城市新建商品住宅价格同比涨幅全部回落、9个城市环比下降或持平
　　5月份，因地制宜、因城施策的房地产调控政策效果继续显现，15个一线和热点二线城市房地产市场基本稳定。从同比看，15个城市新建商品住宅价格涨幅均比上月回落，回落幅度在0.5至6.4个百分点之间。从环比看，9个城市新建商品住宅价格下降或持平；5个城市涨幅在0.5%以内。
　　二、70个大中城市中一二线城市房价同比涨幅持续回落
　　5月份，70个城市中新建商品住宅和二手住宅价格同比涨幅比上月回落的城市分别有29和18个。其中，一二线城市同比涨幅回落尤其明显。据测算，一线城市新建商品住宅和二手住宅价格同比涨幅均连续8个月回落，5月份比4月份分别回落2.2和1.7个百分点；二线城市新建商品住宅和二手住宅价格同比涨幅分别连续6个月和4个月回落，5月份比4月份分别回落0.8和0.5个百分点。
　　三、70个大中城市中房价环比下降及涨幅回落城市个数均有所增加
　　5月份，70个城市中新建商品住宅价格环比下降的城市有9个，比上月增加1个；涨幅回落的城市有26个，比上月增加3个。二手住宅价格环比下降的城市有7个，比上月增加2个；涨幅回落的城市有30个，比上月增加8个。

相似的原文：
 　　国家统计局19日发布数据，5月份，15个一线和热点二线城市新建商品住宅价格同比涨幅全部回落，其中9个城市环比下降或持平。这9个价格环比下降或持平的城市为：北京、上海、南京、杭州、合肥、福州、郑州、深圳、成都。
　　“5月份，因地制宜、因城施策的房地产调控政策效果继续显现，15个一线和热点二线城市房地产市场基本稳定。”国家统计局城市司高级统计师刘建伟说，从同比看，15个城市新建商品住宅价格涨幅均比上月回落，回落幅度在0.5至6.4个百分点之间。从环比看，9个城市新建商品住宅价格下降或持平；5个城市涨幅在0.5%以内。
　　国家统计局当天还发布了5月份70个大中城市住宅销售价格统计数据。刘建伟介绍，5月份，70个大中城市中新建商品住宅和二手住宅价格同比涨幅比上月回落的城市分别有29和18个。其中，一二线城市同比涨幅回落尤其明显。据测算，一线城市新建商品住宅和二手住宅价格同比涨幅均连续8个月回落，5月份比4月份分别回落2.2和1.7个百分点；二线城市新建商品住宅和二手住宅价格同比涨幅分别连续6个月和4个月回落，5月份比4月份分别回落0.8和0.5个百分点。
　　此外，70个大中城市中房价环比下降及涨幅回落城市个数均有所增加。统计显示，5月份，70个大中城市中新建商品住宅价格环比下降的城市有9个，比上月增加1个；涨幅回落的城市有26个，比上月增加3个。二手住宅价格环比下降的城市有7个，比上月增加2个；涨幅回落的城市有30个，比上月增加8个。

