import os
import cv2
import numpy as np
import random
import torch.optim
from torch.utils.data import DataLoader,TensorDataset
from torch import nn
import torch.nn.functional as F
import csv
import time

# 根据三个特征里面有多少个不同的数值，创建直方图
def histogram(clbp,bin_num):
    count=np.zeros((bin_num,1))
    for i in range(clbp.shape[0]):
        for j in range(clbp.shape[1]):
            value=clbp[i][j]
            count[value]+=1
    return count

def CLBP(img):
    h = img.shape[0]
    w = img.shape[1]
    img_mean = np.mean(img)

    CLBP_C = np.zeros((h - 2, w - 2), dtype=np.uint16)
    CLBP_M = np.zeros((h - 2, w - 2), dtype=np.uint16)
    CLBP_S = np.zeros((h - 2, w - 2), dtype=np.uint16)
    code = np.zeros(8, dtype=np.int16)
    for i in range(1, h - 1):
        for j in range(1, w - 1):

            if img[i][j] >= img_mean:
                CLBP_C[i - 1, j - 1] = 1
            else:
                CLBP_C[i - 1, j - 1] = 0

            code[0] = img[i - 1][j - 1]
            code[1] = img[i][j - 1]
            code[2] = img[i + 1][j - 1]
            code[3] = img[i + 1][j]
            code[4] = img[i + 1][j + 1]
            code[5] = img[i][j + 1]
            code[6] = img[i - 1][j + 1]
            code[7] = img[i - 1][j]

            code_uniform = code - img[i][j]
            code_uniform_mean = np.mean(code_uniform)

            code_uniform1 = np.int16(code_uniform >= code_uniform_mean)
            # code_uniform1 = np.zeros(8,dtype=np.int16)
            # for a in range(8):
            #     if code_uniform[a] >= code_uniform_mean:
            #         code_uniform1[a]=1
            #     else:
            #         code_uniform1[a]=0
            CLBP_M[i - 1][j - 1] = int("".join(map(str, code_uniform1)), 2)

            code_uniform2 = np.int16(code_uniform >= 0)
            # code_uniform2 = np.zeros(8,dtype=np.int16)
            # for b in range(8):
            #     if code_uniform[b]>=0:
            #         code_uniform2[b]=1
            #     else:
            #         code_uniform2[b]=0
            CLBP_S[i - 1][j - 1] = int("".join(map(str, code_uniform2)), 2)

    CLBP_S1=histogram(CLBP_S,256).reshape((1,-1))
    CLBP_C1=histogram(CLBP_C,2).reshape((1,-1))
    CLBP_M1=histogram(CLBP_M,256).reshape((1,-1))
    CLBP1 = np.hstack((CLBP_S1, CLBP_C1))
    clbp = np.hstack((CLBP1, CLBP_M1))  # 对三个提取到的特征展平之后再合并
    return clbp

def PCA(X,n):
    def zeroMean(X):  # 零均值化、中心化
        mean=np.mean(X,axis=0)  # 对每一个特征求平均
        new_X=X-mean
        return new_X,mean
    new_X,mean=zeroMean(X)
    # 求协方差矩阵
    cov=np.cov(new_X.T,rowvar=True)  # rowvar=Ture说明每一行数据代表一个样本，如果是False说明每一列数据代表一个样本
    eigVals,eigVects=np.linalg.eig(np.mat(cov))  # 求特征值和特征向量
    eigVals_sort=np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigVals_sort=eigVals_sort[-1:-(n+1):-1]  # 由于上一步是从小到大排序，所以从后往前读，步长为-1，取出最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigVals_sort]
    lowdim_X=new_X*n_eigVect
    return lowdim_X  # 得到低维度的数据

# 进行数据的处理：将图片转为灰度图、尺寸变为256*256，同时将训练集和测试集按照一定的比例分割
def data_process(rate):
    # 将每类图片的名字读取出来
    car = os.listdir('./coursedesign/data/car')
    dog = os.listdir('./coursedesign/data/dog')
    face = os.listdir('./coursedesign/data/face')
    snake = os.listdir('./coursedesign/data/snake')
    # 创建多维的数组存放图片
    car_arr = np.zeros((4000, 256, 256), dtype=np.uint8)  # 只取数据集中的4000张
    dog_arr = np.zeros((4000, 256, 256), dtype=np.uint8)
    face_arr = np.zeros((4000, 256, 256), dtype=np.uint8)
    snake_arr = np.zeros((4000, 256, 256), dtype=np.uint8)
    for i in range(4000):
        img=cv2.imread('./coursedesign/data/car/'+car[i],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        car_arr[i] = img
    for i in range(4000):
        img=cv2.imread('./coursedesign/data/dog/'+dog[i],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        dog_arr[i] = img
    for i in range(4000):
        img=cv2.imread('./coursedesign/data/face/'+face[i],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        face_arr[i] = img
    for i in range(4000):
        img=cv2.imread('./coursedesign/data/snake/'+snake[i],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        snake_arr[i] = img
    # 进行训练集和测试集的分割
    car_train=car_arr[:int(car_arr.shape[0]*rate)]
    car_test=car_arr[int(car_arr.shape[0]*rate):]
    dog_train = dog_arr[:int(dog_arr.shape[0] * rate)]
    dog_test = dog_arr[int(dog_arr.shape[0] * rate):]
    face_train = face_arr[:int(face_arr.shape[0] * rate)]
    face_test = face_arr[int(face_arr.shape[0] * rate):]
    snake_train = snake_arr[:int(snake_arr.shape[0] * rate)]
    snake_test = snake_arr[int(snake_arr.shape[0] * rate):]
    # 设置标签，car-0,dog-1,face-2,snake-3
    car_train_lable=np.full(len(car_train),0, dtype = np.int64)
    car_test_lable = np.full(len(car_test), 0, dtype=np.int64)
    dog_train_lable = np.full(len(dog_train), 1, dtype=np.int64)
    dog_test_lable = np.full(len(dog_test), 1, dtype=np.int64)
    face_train_lable = np.full(len(face_train), 2, dtype=np.int64)
    face_test_lable = np.full(len(face_test), 2, dtype=np.int64)
    snake_train_lable = np.full(len(snake_train), 3, dtype=np.int64)
    snake_test_lable = np.full(len(snake_test), 3, dtype=np.int64)
    # 将每一类的用于训练和测试的图片组合成训练集和测试集，标签同理，并且进行相同的随机打乱
    train_data = np.concatenate([car_train, dog_train, face_train, snake_train])
    np.save("./coursedesign/traindata_clbp100.npy", train_data)
    print("save traindata2.npy done")
    test_data = np.concatenate([car_test, dog_test, face_test, snake_test])
    train_label = np.concatenate([car_train_lable, dog_train_lable, face_train_lable, snake_train_lable])
    test_label = np.concatenate([car_test_lable, dog_test_lable, face_test_lable, snake_test_lable])

    i=np.arange(0,len(train_data))  # 由于需要将训练数据和标签进行同样的打乱，所以这里只能使用索引，并且只需要打乱训练集
    random.shuffle(i)
    train_data = train_data[i]
    train_label = train_label[i]
    return train_data,test_data,train_label,test_label

train_data,test_data,train_label,test_label=data_process(rate=0.875)
train_CLBP_arr=np.zeros((len(train_data),514))  # 通过用一张图片尝试可知最后输出的clbp的维度为514
for i in range(len(train_data)):
    CLBP1=CLBP(train_data[i])  # 提取每一张图像的LBP特征
    print(CLBP1)
    train_CLBP_arr[i]=CLBP1
lowdim_train_CLBP=PCA(train_CLBP_arr,58)  # 用PCA降维
lowdim_train_CLBP1=torch.tensor(lowdim_train_CLBP).to(torch.float32)
train_label1=torch.LongTensor(train_label)
lowdim_train_CLBP2=TensorDataset(lowdim_train_CLBP1,train_label1)  # 将特征和标签拼接,作为神经网络的输入

test_CLBP_arr=np.zeros((len(test_data),514))
for i in range(len(test_data)):
    CLBP2 = CLBP(test_data[i])
    test_CLBP_arr[i] = CLBP2
lowdim_test_CLBP=PCA(test_CLBP_arr,58)
lowdim_test_CLBP1=torch.tensor(lowdim_test_CLBP).to(torch.float32)
test_label1=torch.LongTensor(test_label)
lowdim_test_CLBP2=TensorDataset(lowdim_test_CLBP1,test_label1)

batch_size=1
train_dataloader=DataLoader(lowdim_train_CLBP2,batch_size=batch_size)
test_dataloader=DataLoader(lowdim_test_CLBP2,batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(58, 400),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(400, 200),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(200, 50),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(50, 4)
        )
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return F.sigmoid(logits)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
model=Net()
loss_fn=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
header_train=['trainloss','pred_train','labels_train']
header_test=['pred_test','labels_test']
trainloss=[]
pred_train=[]
labels_train=[]
pred_test=[]
labels_test=[]

def train(dataloader,model,loss_fn,optimizer,num_epochs):
    for epoch in range(num_epochs):
        train_loss = 0.0
        num_correct = 0
        n = 0
        for batch,(X,y) in enumerate(dataloader):
            optimizer.zero_grad()
            X,y=X.to(device), y.to(device)
            pred=model(X)
            loss=loss_fn(pred, y)
            loss.backward()
            train_loss+=loss.item()
            _, pred = torch.max(pred, dim=1)
            pred_train.append(int(pred))
            labels_train.append(int(y))
            if pred==y:
                num_correct+=1
            n+=1
            trainloss.append(train_loss/n)
            optimizer.step()
        print('epoch %d,train loss %.4f, train acc %.3f'%(epoch+1,train_loss/n,num_correct/n))

def test(dataloader,modle):
    modle.eval()
    test_loss=0.0
    n=0
    num_correct=0
    for X,y in dataloader:
        X,y=X.to(device), y.to(device)
        pred = model(X)
        test_loss+=loss_fn(pred,y).item()
        _, pred = torch.max(pred, dim=1)
        pred_test.append(int(pred))
        labels_test.append(int(y))
        if pred==y:
            num_correct+=1
        n+=1
    test_loss/=n
    test_acc=num_correct/n
    print('test loss %.4f, test acc %.3f'%(test_loss,test_acc))

num_epochs=100
train(train_dataloader,model,loss_fn,optimizer,num_epochs)
test(test_dataloader,model)
with open('./coursedesign/datatrain_clbp100.csv','w') as f:
    f_csv=csv.writer(f)
    f_csv.writerow(header_train)
    for i in range(len(trainloss)):
        row = [trainloss[i], pred_train[i], labels_train[i]]
        f_csv.writerow(row)
with open('./coursedesign/datatest_clbp100.csv','w') as f1:
    f_csv1=csv.writer(f1)
    f_csv1.writerow(header_test)
    for i in range(len(pred_test)):
        row = [pred_test[i],labels_test[i]]
        f_csv1.writerow(row)