Thinking1	什么是监督学习，无监督学习，半监督学习。半监督学习介于监督学习和无监督学习之间。通常半监督学习的任务与监督学习一致，即任务中包含有明确的目标（如分类），采用的数据既包括有标签的数据，也包括无标签的数据。作用：只有少量的数据有标签，利用没有标签的数据来学习整个数据的潜在分布，比如标签传播算法label propagation。监督学习（数据集有输入和输出数据）：通过已有的一部分输入数据与输出数据之间的相应关系。生成一个函数，将输入映射到合适的输出，比如分类。无监督学习（数据集中只有输入）：直接对输入数据集进行建模，比如聚类。半监督学习：综合利用有类标的数据和没有类标的数据，来生成合适的分类函数。类别：监督学习分为分类和回归：最广泛被使用的分类器有人工神经网络、支持向量机、近期邻居法、高斯混合模型、朴素贝叶斯方法、决策树和径向基函数分类。回归：线性回归，神经网络。无监督学习：主要由聚类。
Thinking2	K-means中的k值如何选取：手肘法选K，多选几次，然后画出loss值的趋势图，选斜率最大的点。
Thinking3	随机森林采用了bagging集成学习，bagging指的是什么：装袋有随机的放回的方法。多数投票机制，Bagging流派，各分类器之间没有依赖关系，可各自并行，比如随机森林（Random Forest）
Thinking4	主动学习和半监督学习的区别是什么：主动学习是如果机器可以自己选择学习的样本，可以使用较少的训练取得更好的效果。需要人工介入，模型主动向工作者提供数据。半监督学习指在训练数据十分稀少的情况下，利用没有标签的数据，提高模型效果的方法。表征学习：也称为特征学习，目的是对复杂的原始数据化繁为简，把原始数据的无效信息剔除，有效信息更有效地进行提炼，形成特征，如果特征被有效提取，那么之后的学习任务会更简单和精确。