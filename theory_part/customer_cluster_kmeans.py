# coding: utf-8
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

# 数据加载
data = pd.read_csv('Mall_Customers.csv', encoding='gbk')
train_x = data[["Gender","Age","Annual Income (k$)", "Spending Score (1-100)"]]

# LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_x['Gender'] = le.fit_transform(train_x['Gender'])

# 规范化到 [0,1] 空间
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
pd.DataFrame(train_x).to_csv('temp.csv', index=False)
#print(train_x)
# 使用KMeans聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
# 合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'聚类结果'},axis=1,inplace=True)
print(result)
# 将结果导出到CSV文件中
result.to_csv("customer_cluster_result11.csv",index=False)

#以下为结果部分
zhangyuxi@ZhangdeMacBook-Pro cluster % python3 customer_cluster_kmeans.py
customer_cluster_kmeans.py:14: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  train_x['Gender'] = le.fit_transform(train_x['Gender'])
/usr/local/lib/python3.7/site-packages/sklearn/preprocessing/data.py:334: DataConversionWarning: Data with input dtype int64 were all converted to float64 by MinMaxScaler.
  return self.partial_fit(X, y)
     CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)  聚类结果
0             1    Male   19                  15                      39     2
1             2    Male   21                  15                      81     2
2             3  Female   20                  16                       6     0
3             4  Female   23                  16                      77     0
4             5  Female   31                  17                      40     0
..          ...     ...  ...                 ...                     ...   ...
195         196  Female   35                 120                      79     0
196         197  Female   45                 126                      28     0
197         198    Male   32                 126                      74     2
198         199    Male   32                 137                      18     1
199         200    Male   30                 137                      83     2

[200 rows x 6 columns]