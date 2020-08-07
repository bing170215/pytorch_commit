import numpy as np

data = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
y = np.array([1,2,3,4,5])

print('-------第1种方法：通过打乱索引从而打乱数据,好处是1:数据量很大时能够节约内存,2每次都不一样----------')
data = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
data_num, _= data.shape #得到样本数  
index = np.arange(data_num) #生成下标  
np.random.shuffle(index)
print('-------原数据：----------')
print('数据：',data)
print('标签：', y)
print('-------打乱数据：----------')
print('数据：',data[index])
print('标签：',y[index])

print('-------第2种方法：直接的打乱数据,利用随机数种子，好处：每次打乱的顺序是固定的----------')
data = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
y = np.array([1,2,3,4,5])

print('-------原数据：----------')
print('数据：',data)
print('标签：', y)
print('-------打乱数据：----------')
np.random.seed(116)
np.random.shuffle(data)
np.random.seed(116)
np.random.shuffle(y)
print('数据：',data)
print('标签：', y)