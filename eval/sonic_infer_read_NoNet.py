import torch
from utils_infer import *
import matlab
import matlab.engine
import time
import torch.nn as nn
from calError import calError
import os

"""
推理的时候，需要修改的数据：
1.sir 
2.ResPath
3.labelPath
"""

#####################################################
sir = 20  # 信号的信噪比
path = "../../../AllSnrModel/BigData/MoreLayerTCNMSE_SmoothL1/"
is_validate = False
groupNum = 5
# resPath = f"Result_NoNet/snr_{snr}/"

#####################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_sonic_4D(sonic):
    spl = np.zeros((1, 3, 4, 300))
    for i in range(3):
        spl[0, i, :, :] = sonic[:, i * 300:(i + 1) * 300]
    return spl

def plot_singal_all_output(data):
    x, y = data.shape
    for i in range(x):
        length = y
        xp1 = np.linspace(1, length, length)
        yp1 = data[i, :]
        plt.subplot(2, 2, i + 1)
        plt.plot(xp1, yp1, color='blue')
        plt.title("第{}通道".format(i + 1))

    plt.show()


def writefile_tensor(res_sig):
    x, y = res_sig.shape  # 4,2700
    with open(path+"testdata.txt", 'w') as f:
        for num in range(y):
            f.write(str(res_sig[0, num]) + ' ' + str(res_sig[1, num]) + ' ' +
                    str(res_sig[2, num]) + ' ' + str(res_sig[3, num]) + '\n')


def writelabel_tensor(labelsig):
    x, y = labelsig.shape  # 4*300
    with open(path+"testlabel.txt", 'w') as f:
        for num in range(y):
            f.write(str(labelsig[0, num]) + ' ' + str(labelsig[1, num]) + ' ' +
                    str(labelsig[2, num]) + ' ' + str(labelsig[3, num]) + '\n')


def infer(input,model):
    """
    :param input: 4*900
    :param model: model
    :return:
    """
    input_t = torch.tensor(input,dtype=torch.float32)  # 4*900
    input_t = input_t.to(device)
    with torch.no_grad():
        output = torch.zeros((1,4,900))
        for k in range(4):
            ip = torch.unsqueeze(torch.unsqueeze(input_t[k,:],dim=0),dim=0)  # 1,1,900
            output[0,k,:] = model(ip)

        output = output.cpu().numpy()
        return output


def splitLabel(label):
    lbs = label.strip("\n").split(" ")
    for idx,l in enumerate(lbs):
        lbs[idx] = float(l)

    return lbs

def splitAndInfer(data):
    stripped_list = [s.strip("\n") for s in data]
    npData = np.zeros((4,2700))
    idx = 0
    for line in stripped_list:
        numlist = line.split(" ")
        npData[:,idx] = numlist
        idx += 1

    return npData


def makeFloat(line):
    ldata = line.strip("\n")
    ldata = ldata.split(" ")
    for i in range(len(ldata)):
        ldata[i] = float(ldata[i])

    return ldata

def split_sonic(sonic):
    """
    :param sonic: origin sonic 4*900
    :return: sonic in shape of 3*4*300
    """
    spl = np.zeros((3, 4, 300))
    for i in range(3):
        spl[i, :, :] = sonic[:, i * 300:(i + 1) * 300]

    return spl


def clearTXT(path):
    file = open(path+"error.txt", 'w').close()
    # file = open(path+"testdata.txt", 'w').close()
    file = open(path+"calError.txt", 'w').close()


if __name__ == '__main__':
    print("============================")
    print("loading matlab environment")
    eng = matlab.engine.start_matlab()
    print("matlab environment done")
    while sir >= 0:
        resPath = f"Result_SingleMusic/sir_{sir}/"
        print("----------------------")
        print("snr:",sir)
        print("----------------------")
        if not os.path.exists(resPath):
            os.makedirs(resPath)

        fileCount = 0
        clearTXT(resPath)
        txtCount = 100
        for count in range(txtCount):
            print("====================")
            print("count:",count)
            labelpath = f"ValidateSIR/sir_{sir}/label/label_{count}.txt"
            datapath = f"ValidateSIR/sir_{sir}/data/data_{count}.txt"
            s = time.time()
            with open(labelpath,'r') as f:
                labellines = f.readlines()
            with open(datapath,'r') as ff:
                datalines = ff.readlines()

            for i in range(groupNum):
                data = datalines[i*2700:(i+1)*2700]
                label = labellines[i]
                labels = splitLabel(label)
                output = splitAndInfer(data)
                # plot_singal_all_output(output)
                writefile_tensor(output)
                print(" ")
                print(f"realTheta:{labels[0]}  realFai:{labels[1]}")
                eng.pyMatlibMusic(float(labels[0]), float(labels[1]),path,resPath)
                e = time.time()
                print("time:",e-s)

        calError(resPath)
        sir -= 1