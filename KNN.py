# _*_ coding: utf-8 _*_ 
'''
Created on Month 22, 2018
kNN: k Nearest Neighbors

Input:      TargetSet: vector to compare to existing dataset (1xN)
            RefDataSet: size m data set of known vectors (NxM)
            Labels: data set labels (1xM vector)
            K: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: liupu
'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import operator

# 创建分类器函数
def Classification(TargetSet, RefDataSet, Labels, K):
    DataSetSize = dataSet.shape[0]
    Distances = (((RefDataSet - TargetSet) ** 2).sum(axis = 1)) ** 0.5
    SortedDistIndicies = Distances.argsort()     
    ClassCount={}          
    for i in range(k):
        VoteIlabel = Labels[SortedDistIndicies[i]]
        ClassCount[VoteIlabel] = ClassCount.get(VoteIlabel,0) + 1
    SortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return SortedClassCount[0][0]

# 创建数据集
def CreateDataSet():
    group = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    labels = ['Love','Love','Love','Action','Action','Action']
    return group,labels

# 创建数据导入函数
def DataInput(filename):
    DatingTestSet = pd.read_table(filename,header = None)
    PriTestSet = DatingTestSet.iloc[:,[0,1,2]]
    PriTestSet.columns = ['Flying','Game','Icecream']
    LabelTestSet = DatingTestSet.iloc[:,3]
    return PriTestSet,LabelTestSet

# 创建归一化函数
def NormValue(dataset):
    NormDataSet = np.zeros(dataset.shape)
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    rangeVals = maxVals - minVals
    m = dataset.shape[0]
    NormDataSet = (dataset - minVals) / rangeVals
    return NormDataSet,rangeVals,minVals

# 创建分类测试函数  
def datingClassTest():
    TestRatio = 0.10      #hold out 10%
    PriTestSet,LabelTestSet = DataInput('DatingTestSet.txt')       #load data setfrom file
    NormData, ranges, minVals = NormValue(PriTestSet)
    m = NormData.shape[0]
    NumofTestSet = int(m*TestRatio)
    errorCount = 0.0
    for ii in range(NumofTestSet):
        classifyResult = Classification(NormData.loc[ii],NormData.loc[NumofTestSet:m],LabelTestSet[NumofTestSet:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifyResult, LabelTestSet[ii]))
        if (classifyResult != LabelTestSet[ii]): 
        	errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(NumofTestSet)))
    print(errorCount)

# 创建图像向量化函数
def image2vector(filename):
    BinaryVector = np.zeros(1024)
    ImageMatrix = pd.read_table(filename,header=None)
    for ii in range(32):
        for jj in range(32):
            BinaryVector[32*ii+jj] = int(np.array(ImageMatrix.loc[0])[0][0])
    return BinaryVector

# 创建手写数字识别程序
def handwritingClass():
    hwLabels = [] 
    TrainingFileNameList = os.listdir('./digits/trainingDigits')
    NumofTrainingFile = len(TrainingFileNameList)
    TrainingMat = np.zeros((NumofTrainingFile,1024))
    for ii in range(NumofTrainingFile):
        FileName = TrainingFileNameList[ii]
        FileNameSplit = FileName.split('.')[0]
        ClassFile = int(FileNameSplit.split('_')[0])
        hwLabels.append(ClassFile)
        TrainingMat[ii,:] = image2vector('./digits/trainingDigits/%s' %FileName)
    TestFileNameList = os.listdir('./digits/testDigits')
    NumofTestFile = len(TestFileNameList)
    errorcount = 0
    for jj in range(NumofTestFile):
        FileName = TestFileNameList[jj]
        FileNameSplit = FileName.split('.')[0]
        ClassFile = int(FileNameSplit.split('_')[0])
        TestVector = image2vector('./digits/testDigits/%s' %FileName)
        ClassfyResult = classification(TestVector,TrainingMat,hwLabels,3)
        print('The Result of ClassFy is:%d,the real answer is %d' %(ClassfyResult,ClassFile))
        if ClassfyResult != ClassFile:
            errorcount += 1
    print('The total error is %d' %errorcount)
    print('The total error rate is: %f' %(errorcount/NumofTestFile))