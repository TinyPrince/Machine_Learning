# _*_ coding: utf-8 _*_ 
'''
@author: liupu
'''
import numpy as np
import operator
import matplotlib.pyplot as plt 

# 定义KNN分类算法函数
def classification(TargetData,RefDataSet,labels,k):
	'''
	'''
    RefDataSetSize = RefDataSet.shape[0]
    DiffMat = np.tile(TargetData,(RefDataSetSize,1)) - RefDataSet
    SquarDiffMat = DiffMat ** 2
    SumSquareDiffMat = SquarDiffMat.sum(axis = 1)
    Distance = SumSquareDiffMat ** 0.5
    SortedDistIndicies = Distance.argsort()
    ClassCount = {}
    for ii in range(k):
        VoteIndiviceLabel = labels[SortedDistIndicies[ii]]
        ClassCount[VoteIndiviceLabel] = ClassCount.get(VoteIndiviceLabel,0) + 1
        SortedClassCount = sorted(ClassCount.items(),
                                  key = operator.itemgetter(1),reverse=True)
        return SortedClassCount[0][0]

# 创建数据集
def createDataSet():
    group = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    labels = ['Love','Love','Love','Action','Action','Action']
    return group,labels

# 绘制训练集数据分布
plt.scatter(group[:,0],group[:,1])
plt.xlabel('Action')
plt.ylabel('Kiss')
plt.show()

# 运用分类函数对未知样本进行分类
classification([0,0],group,labels,3)