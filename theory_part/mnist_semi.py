import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

# 数据加载
digits = datasets.load_digits()
# 第一个数字
print(digits.data[0])
print(digits.target[0])
# 全部数据
X = digits.data
y = digits.target

n_total_samples = len(digits.data) # 1797
n_labeled_points = int(n_total_samples*0.1) #179

# 创建LR分类器
lr = LogisticRegression()
lr.fit(X[:n_labeled_points], y[:n_labeled_points])
# 对剩余90%数据进行预测
predict_y=lr.predict(X[n_labeled_points:])
true_y = y[n_labeled_points:] 
print("准确率", (predict_y == true_y).sum()/(len(true_y)))
print("-"*20)

# 使用半监督学习
# 复制一份y
y_train = np.copy(y)
# 把未标注的数据全部标记为-1，也就是后90%数据
y_train[n_labeled_points:] = -1 

# 使用标签传播模型，进行训练
lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5) 
lp_model.fit(X,y_train)
# 得到预测的标签
predict_y = lp_model.transduction_[n_labeled_points:] 
# 真实的标签
true_y = y[n_labeled_points:] 
print("预测标签", predict_y)
print("真实标签", true_y)
print("准确率", (predict_y == true_y).sum()/(len(true_y)))
cm = confusion_matrix(true_y, predict_y, labels = lp_model.classes_)
print("Confusion matrix", cm)
#以下为结果部分
[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
 15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
  0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
  0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
0
/usr/local/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/usr/local/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
准确率 0.8182941903584673
--------------------
/usr/local/lib/python3.7/site-packages/sklearn/semi_supervised/label_propagation.py:292: ConvergenceWarning: max_iter=5 was reached without convergence.
  category=ConvergenceWarning
预测标签 [0 2 2 ... 8 9 8]
真实标签 [0 2 2 ... 8 9 8]
准确率 0.915327564894932
Confusion matrix [[159   0   0   0   0   0   0   0   0   0]
 [  0  94   5   0   0   0  49   0  15   1]
 [  0   0 159   1   0   0   0   0   1   0]
 [  0   0   0 161   0   0   0   0   1   4]
 [  0   0   0   0 155   0   3   3   0   3]
 [  0   0   0   0   0 152   1   0   0   9]
 [  0   0   0   0   0   0 164   0   0   0]
 [  0   0   0   0   0   0   0 160   1   0]
 [  0   3   1   1   0   0   0   0 148   3]
 [  0  20   0   1   0   8   0   0   3 129]]