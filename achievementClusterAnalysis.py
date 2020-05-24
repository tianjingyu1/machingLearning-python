import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadData(filePath):
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    averageScore = []  #均分
    dormitoryNumber = []  #宿舍号
    studentNumber = []  #学号
    gender = []  #性别
    achievement = []  #单科成绩
    rank = [] #排名
    for line in lines:
        items = line.strip().split(",")
        studentNumber.append(items[0])
        dormitoryNumber.append(items[1])
        gender.append(items[2])
        achievement.append(items[3])  #单科成绩，除均分，排名几项等外都可以
        averageScore.append([float(items[13])]) #13、25、37、48、50  分别为前四次均分与总均分
        #averageScore = [[i] for i in averageScore]  # 把每个样本变成一个单独的‘向量’就可以了。
        rank.append(items[14])  #14、26、38、49、51  分别为前四次排名与总排名
    return studentNumber,dormitoryNumber,gender,achievement,averageScore,rank


if __name__ == '__main__':
    studentNumber,dormitoryNumber,gender,achievement,averageScore,rank = loadData('四学期成绩汇总.txt')
    km = KMeans(n_clusters=9)
    #averageScore = np.array(averageScore).reshape(1, -1)
    #print(averageScore)
    label = km.fit_predict(averageScore)
    averageScoreShow = np.sum(km.cluster_centers_, axis=1)
    dormitoryNumberShow = [[], [], [], [],[],[],[],[],[]]
    for i in range(len(dormitoryNumber)):
        dormitoryNumberShow[label[i]].append(dormitoryNumber[i])
    for i in range(len(dormitoryNumberShow)):
        print("averageScore:%.2f" % averageScoreShow[i])
        print(dormitoryNumberShow[i])
