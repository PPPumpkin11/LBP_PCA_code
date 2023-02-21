import os
import cv2
import numpy as np
import random
import torch.optim
from torch.utils.data import DataLoader,TensorDataset
from torch import nn
import torch.nn.functional as F
import csv

# 计算出LBP之后，要确定将这个LBP对应到59类中的哪一类
def gen_fix_vec():
    l=[]
    a=-1
    for i in range(256):
        bit='{:08b}'.format(i)
        compare=bit[0]
        count=0
        for j in range(1,len(bit)):
            if bit[j] != compare:
                count+=1
                compare=bit[j]
        if count<=2:
            a+=1
            l.append(i)  # 如果跳变次数小于2，那么按照顺序放到l里面
    return l

def sort_to58(L, LBP):
    for k in range(len(L)):
        if L[k]==LBP:
            return k

def uniform_LBP(img):
    L = gen_fix_vec()

    h=img.shape[0]
    w=img.shape[1]
    dst=np.zeros((h-2,w-2),dtype=img.dtype)  # 新建一张图来存放每一个像素点最后那个编码对应的十进制数
    for i in range(1,h-1):
        for j in range(1,w-1):
            center=img[i][j]
            code=[]
            if img[i-1][j-1]>=center:
                code.append(1)
            else:
                code.append(0)
            if img[i-1][j]>=center:
                code.append(1)
            else:
                code.append(0)
            if img[i-1][j+1]>=center:
                code.append(1)
            else:
                code.append(0)
            if img[i][j+1]>=center:
                code.append(1)
            else:
                code.append(0)
            if img[i][j-1]>=center:
                code.append(1)
            else:
                code.append(0)
            if img[i+1][j+1]>=center:
                code.append(1)
            else:
                code.append(0)
            if img[i+1][j]>=center:
                code.append(1)
            else:
                code.append(0)
            if img[i+1][j-1]>=center:
                code.append(1)
            else:
                code.append(0)
            LBP=code[0]+code[1]*2+code[2]*4+code[3]*8+code[4]*16+code[5]*32+code[6]*64+code[7]*128
            compare=code[0]  # 设置一个变量存放用来与下一个数对比的数
            count=0  # 设置变量存放跳变次数
            for a in range(1,len(code)):
                if code[a] != compare:
                    count+=1
                    compare=code[a]  # 如果发生了跳变，将发生跳变的数设置为用来对比的数
            if count>2:
                dst[i-1][j-1]=58  # 如果跳变次数大于2就归入第59类
            else:
                dst[i - 1][j - 1] = sort_to58(L, LBP)
    return dst

# 参考画直方图的方法，得出lbp特征
def histogram(a):
    rate = np.zeros(59)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            rate[a[i, j]] += 1
    return rate

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

    test_data = np.concatenate([car_test, dog_test, face_test, snake_test])

    train_label = np.concatenate([car_train_lable, dog_train_lable, face_train_lable, snake_train_lable])
    test_label = np.concatenate([car_test_lable, dog_test_lable, face_test_lable, snake_test_lable])

    i=np.arange(0,len(train_data))  # 由于需要将训练数据和标签进行同样的打乱，所以这里只能使用索引，并且只需要打乱训练集
    random.shuffle(i)
    train_data = train_data[i]
    train_label = train_label[i]
    return train_data,test_data,train_label,test_label

train_data,test_data,train_label,test_label=data_process(rate=0.875)  # 3500张图片用来训练，500张用来测试
# 对训练集和测试集提取lbp特征
train_LBP_arr=np.zeros((len(train_data),59))
for i in range(len(train_data)):
    LBP=histogram(uniform_LBP(train_data[i]))  # 提取每一张图像的LBP特征
    print(LBP)
    train_LBP_arr[i]=LBP
lowdim_train_LBP=PCA(train_LBP_arr,49)  # 用PCA降维
lowdim_train_LBP1=torch.tensor(lowdim_train_LBP).to(torch.float32)
train_label1=torch.LongTensor(train_label)
lowdim_train_LBP2=TensorDataset(lowdim_train_LBP1,train_label1)  # 将特征和标签拼接,作为神经网络的输入

test_LBP_arr=np.zeros((len(test_data),59))
for i in range(len(test_data)):
    LBP = histogram(uniform_LBP(test_data[i]))
    test_LBP_arr[i] = LBP
lowdim_test_LBP=PCA(test_LBP_arr,49)
lowdim_test_LBP1=torch.tensor(lowdim_test_LBP).to(torch.float32)
test_label1=torch.LongTensor(test_label)
lowdim_test_LBP2=TensorDataset(lowdim_test_LBP1,test_label1)

batch_size=1
train_dataloader=DataLoader(lowdim_train_LBP2,batch_size=batch_size)
test_dataloader=DataLoader(lowdim_test_LBP2,batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(49, 400),

            nn.ReLU(),

            nn.Linear(400, 200),

            nn.ReLU(),

            nn.Linear(200, 50),

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
with open('./coursedesign/datatrain_100_pca.csv1','w') as f:
    f_csv=csv.writer(f)
    f_csv.writerow(header_train)
    for i in range(len(trainloss)):
        row = [trainloss[i], pred_train[i], labels_train[i]]
        f_csv.writerow(row)
with open('./coursedesign/datatest_100_pca.csv1','w') as f1:
    f_csv1=csv.writer(f1)
    f_csv1.writerow(header_test)
    for i in range(len(pred_test)):
        row = [pred_test[i],labels_test[i]]
        f_csv1.writerow(row)
