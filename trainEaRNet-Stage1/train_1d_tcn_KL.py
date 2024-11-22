from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import numpy as np
import matplotlib.pyplot as plt
from DRSN_MoreLayer_tcn import *

device = torch.device("cuda")
num_per_gruop = 900  # 定义一组数据总共有几行
batchsize = 32
alpha_mse = 0.5
beta_kl = 0.5
pretrained = False
opt_pretrained = False

def split_sonic_SingleChannel(sonic):
    """
    :param sonic: origin sonic 4*900
    :return: List : 1*900,1*900,1*900,1*900
    """
    # spl = np.zeros((3, 4, 300))
    spl = np.zeros((1,900))
    spl_list = []
    for i in range(4):
        spl[:,:] = sonic[i,:]
        spl_list.append(spl)

    for i in range(4):  # Convert to Tensor
        spl_list[i] = torch.tensor(spl_list[i],dtype=torch.float32)

    return spl_list


def Norm2One(a, reference):
    a = a / reference
    return a


def makeFloat(line):
    ldata = line.strip("\n")
    ldata = ldata.split(" ")
    for i in range(len(ldata)):
        ldata[i] = float(ldata[i])

    return ldata

def makePlot_loss(trainLoss, testLoss):
    # 画出曲线
    loss_len = len(trainLoss)
    x_loss = np.arange(0, loss_len, 1)
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x_loss, trainLoss, linewidth=1, linestyle="solid", label="trainLoss")
    plt.plot(x_loss, testLoss, linewidth=1, linestyle="solid", label="testLoss")
    plt.legend()
    plt.title('Loss curve')
    plt.savefig("Loss.png")
    plt.close()

def makePlot_mseLoss(trainMseLoss, testMseLoss):
    # 画出曲线
    loss_len = len(trainMseLoss)
    x_loss = np.arange(0, loss_len, 1)
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x_loss, trainMseLoss, linewidth=1, linestyle="solid", label="trainMseLoss")
    plt.plot(x_loss, testMseLoss, linewidth=1, linestyle="solid", label="testMseLoss")
    plt.legend()
    plt.title('MseLoss curve')
    plt.savefig("MseLoss.png")
    plt.close()


def makePlot_KLLoss(trainKLLoss, testKLLoss):
    # 画出曲线
    loss_len = len(trainKLLoss)
    x_loss = np.arange(0, loss_len, 1)
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x_loss, trainKLLoss, linewidth=1, linestyle="solid", label="trainKLLoss")
    plt.plot(x_loss, testKLLoss, linewidth=1, linestyle="solid", label="testKLLoss")
    plt.legend()
    plt.title('KLLoss curve')
    plt.savefig("KLLoss.png")
    plt.close()



class KL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input,target):
        diff = input - target  # diff 是batchsize*1*900
        aa,bb,cc = diff.size()
        std = torch.std(diff, dim=2)
        mean = torch.mean(diff, dim=2)
        total_loss = -0.5*(1 + torch.log(torch.pow(std, 2)) - torch.pow(std, 2) - torch.pow(mean, 2))
        loss = torch.sum(total_loss) / (aa*bb*cc)
        return loss



class SonicData(Dataset):
    def __init__(self, is_train, transform=None):
        self.train_set = self.load_data(is_train)
        self.transform = transform

    def __getitem__(self, index):
        Sonic_data, Sonic_label = self.train_set[index]
        return Sonic_data, Sonic_label

    def __len__(self):
        return len(self.train_set)

    def load_data(self, is_train):
        print("loading data")
        if is_train:
            print("导入训练集")
            epoch = 90
            label_path = "../UpSampleSnrAllData/label/train/label_"
            data_path = "../UpSampleSnrAllData/data/train/data_"
        else:
            print("导入测试集")
            epoch = 10
            label_path = "../UpSampleSnrAllData/label/test/label_"
            data_path = "../UpSampleSnrAllData/data/test/data_"

        total_data = []
        singal_data = []
        line_data = []
        line_label = []
        lineCount = 0
        for count in range(epoch):  # 总共100个文件，每个文件里面有500组数据，一组数据300行
            start = time.time()
            print("第{}轮导入数据".format(count + 1))
            if is_train:
                Dcount = count
            else:
                Dcount = count + 90

            batchCount = 0  # 每次循环需要进行清零操作
            with open(label_path + str(Dcount) + ".txt") as fline:
                print(label_path + str(Dcount) + ".txt")
                total_label_line = fline.readlines()

            path = data_path + str(Dcount) + ".txt"
            print(path)
            with open(path) as f:
                for index,data_line in enumerate(f.readlines()):  # f.readlines表示读取了所有行数据
                    label_line = total_label_line[index]
                    lbl = makeFloat(label_line)
                    line_label.append(lbl)
                    ldata = makeFloat(data_line)
                    line_data.append(ldata)
                    lineCount += 1
                    if lineCount % num_per_gruop == 0 and lineCount != 0:  # 900行为一批数据
                        lineCount = 0
                        # 先把line_data转化为numpy格式，然后进行转置，
                        line_data_np = np.asarray(line_data)  # 900*4
                        line_data = []
                        line_data_np = line_data_np.T  # 4*900,
                        ListData = split_sonic_SingleChannel(line_data_np)  # List :4,1*900

                        # for i in range(len(ListData)):
                        #     singal_data.append(ListData[i])
                        # cycleData_tensor = torch.tensor(cycleData, dtype=torch.float32)  # 转tensor格式
                        # singal_data.append(cycleData_tensor)  # single_data表示单单一批数据，包括了原数据与标签

                        line_label_np = np.asarray(line_label)  # 900*4
                        line_label = []
                        line_label_np = line_label_np.T  # 900*4
                        ListLabel = split_sonic_SingleChannel(line_label_np)  # List :4,1*900

                        # for i in range(len(ListLabel)):
                        #     singal_data.append(ListLabel[i])
                        # cycleLabel_tensor = torch.tensor(cycleLabel, dtype=torch.float32)  # 转tensor格式
                        # singal_data.append(cycleLabel_tensor)  # single_data表示单单一批数据，包括了原数据与标签
                        for i in range(len(ListLabel)):
                            singal_data.append(ListData[i])
                            singal_data.append(ListLabel[i])
                            total_data.append(singal_data)
                            singal_data = []

                        # total_data.append(singal_data)
                        singal_data = []
                        batchCount += 1

            end = time.time()
            print("第{}轮导入数据，耗时{}".format(count + 1, end - start))

        print("----------完成导入数据，开始预处理----------")
        return total_data


print("loading data")
start = time.time()
train_set = SonicData(is_train=True)
end = time.time()
print("train_set successfully loaded, using time:{}".format(end - start))

start = time.time()
test_set = SonicData(is_train=False)
end = time.time()
print("test_set successfully loaded, using time:{}".format(end - start))

print("train_set length:", len(train_set))
print("test_set length:", len(test_set))

train_dataLoader = DataLoader(train_set, batch_size=batchsize)
test_dataLoader = DataLoader(test_set, batch_size=batchsize)

SncNet = NestedUNet_DRSN1d_MoreLayer(num_classes=1,input_channels=1).to(device)

if pretrained:
    print("loading pretrained model")
    pretrained_model = torch.load("pretrained.pth")
    SncNet.load_state_dict(pretrained_model,strict=False)

learning_rate = 0.0001
optimizer = torch.optim.Adam(SncNet.parameters(), lr=learning_rate)
if opt_pretrained:
    print("loading pretrained optimizer")
    pretrained_opt = torch.load("Adam_pretrained.pth")
    optimizer.load_state_dict(pretrained_opt)

loss_func = nn.MSELoss()
loss_func = loss_func.to(device)
kl_func = KL_Loss()
kl_func = kl_func.to(device)

# 可以试试Adamw
# 设置训练参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
net_epoch = 150
# 利用tensorboard观察准确度与损失度

total_trainLoss = []
total_testLoss = []
total_trainMseLoss = []
total_testMseLoss = []
total_trainKLLoss = []
total_testKLLoss = []

Add_total = False
min_loss = 99999
min_epoch = 0

# 所有东西保持不变，加入KL散度，让误差保持正态分布
# Loss公式来自VAE论文，std是标方差，mu是均值，N是元素数量，KL_Loss加上MSE的loss，在做反向传播

# 传入误差图
for i in range(net_epoch):
    start = time.time()
    print("--------第{}轮训练开始--------".format(i + 1))
    SncNet.train()
    # 每次循环都要清零
    train_loss = 0
    train_mse_loss = 0
    train_kl_loss = 0
    train_loss_value = 0
    mse_loss_value = 0
    KL_loss_value = 0
    train_step = 0
    output_txt = ""
    output_txt += "--------第{}轮训练开始--------\n".format(i + 1)
    for data in train_dataLoader:
        train_step += 1
        Sonics, labels = data
        Sonics = Sonics.to(device)
        labels = labels.to(device)
        # 想要使用to(device)函数，就必须要求他们为tensor类型
        optimizer.zero_grad()  # 优化器清楚梯度
        output = SncNet(Sonics)

        mse_loss_value = loss_func(output, labels)
        kl_loss_value = kl_func(output, labels)

        train_loss_value = alpha_mse*mse_loss_value + beta_kl*kl_loss_value
        train_loss_value.backward()  # 反向传播

        optimizer.step()
        total_train_step += 1

        train_mse_loss += mse_loss_value.item()
        train_kl_loss += kl_loss_value.item()
        train_loss += train_loss_value.item()


    avg_train_mse_loss = train_mse_loss/train_step
    avg_train_kl_loss = train_kl_loss/train_step
    avg_trainloss = train_loss/train_step
    print("第{}轮  训练集mse_loss={}  kl_loss={}  train_loss={}".format(i + 1, avg_train_mse_loss,avg_train_kl_loss,avg_trainloss))
    output_txt += "第{}轮  训练集mse_loss={}  kl_loss={}  train_loss={} \n".format(i + 1, avg_train_mse_loss,avg_train_kl_loss,avg_trainloss)
    total_trainLoss.append(avg_trainloss)
    total_trainMseLoss.append(avg_train_mse_loss)
    total_trainKLLoss.append(avg_train_kl_loss)

    SncNet.eval()  # 这句话需要写上去，因为会对dropout，batchnorm产生影响
    # 这里产生了个问题，就是每次训练结束后如何知道我训练出来的结果达到了我想要的结果，因此这时候就需要测试集，并且这一过程不需要调优了
    test_loss = 0
    test_mse_loss = 0
    test_kl_loss = 0
    test_loss_value = 0
    mse_loss_value = 0
    kl_loss_value = 0
    test_step = 0
    with torch.no_grad():  # 这句话是让梯度不参与训练
        for data in test_dataLoader:
            test_step += 1
            Sonics, labels = data
            Sonics = Sonics.to(device)
            labels = labels.to(device)
            # outputs = SncNet(Sonics)
            # test_loss_value = loss_func(outputs, labels)  # 需要注意的是，loss_value是一个Tensor数据类型
            output = SncNet(Sonics)
            mse_loss_value = loss_func(output, labels)  # 计算损失值
            kl_loss_value = kl_func(output, labels)
            test_loss_value = alpha_mse*mse_loss_value + beta_kl*kl_loss_value

            # 计算损失值与准确度
            total_test_step += 1

            test_mse_loss += mse_loss_value.item()
            test_kl_loss += kl_loss_value.item()
            test_loss += test_loss_value.item()

    avg_test_mse_loss = test_mse_loss/test_step
    avg_test_kl_loss = test_kl_loss/test_step
    avg_testloss = test_loss/test_step


    print("第{}轮  测试集mse_value={}  kl_loss={}  test_loss={} ".format(i + 1,avg_test_mse_loss,avg_test_kl_loss, avg_testloss))
    output_txt+="第{}轮  测试集mse_value={}  kl_loss={}  test_loss={} \n".format(i + 1,avg_test_mse_loss,avg_test_kl_loss, avg_testloss)
    total_testLoss.append(avg_testloss)
    total_testMseLoss.append(avg_test_mse_loss)
    total_testKLLoss.append(avg_test_kl_loss)

    end = time.time()
    print("总共用时：", end - start)
    output_txt += "总共用时:{} \n".format(end - start)

    avgloss = (avg_trainloss+avg_testloss)/2
    print("第{}轮  平均损失avgloss={}  ".format(i + 1, avgloss))
    output_txt += "第{}轮  平均损失avgloss={}  \n".format(i + 1, avgloss)

    if avgloss < min_loss:
        print("reload best.pth")
        min_epoch = i
        min_loss = avgloss
        torch.save(SncNet.state_dict(), "best.pth")
        with open("bestpthInformation.txt",'w') as f:
            f.write("epoch:{}\n"
                    "train_mse_loss={}  train_kl_loss={}  train_loss={}\n"
                    "test_mse_loss={}   test_kl_loss={}   test_loss={}\n"
                    "avg_loss={}\n".format(
                i + 1,
                avg_train_mse_loss,avg_train_kl_loss,avg_trainloss,
                avg_test_mse_loss,avg_test_kl_loss,avg_testloss,
                avgloss))


    with open("train_info.txt",'a') as f_t:
        f_t.write(output_txt)

    torch.save(SncNet.state_dict(), "last.pth")
    makePlot_loss(total_trainLoss, total_testLoss)
    makePlot_KLLoss(total_trainKLLoss,total_testKLLoss)
    makePlot_mseLoss(total_trainMseLoss,total_testMseLoss)

