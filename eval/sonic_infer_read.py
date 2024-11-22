import torch
from utils_infer import *
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

N = ["DRSNMoreLayer","MoreLayerTCNMSE","DRSNMoreLayerMSE","LayerMoreTCNMSE","MoreLayerTCNMSE_SmoothL1"]
#####################################################
snr = 10  # 信号的信噪比
path = "../../AllSnrModel/BigData/MoreLayerTCNMSE_SmoothL1/"
# resPath = f"Result/snr_{snr}/"
_Net = N[1]
is_validate = False
groupNum = 5

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
    a,x, y = res_sig.shape  # 1,4,900
    with open(path+"testdata.txt", 'w') as f:
        for num in range(y):
            f.write(str(res_sig[0,0, num]) + ' ' + str(res_sig[0,1, num]) + ' ' +
                    str(res_sig[0,2, num]) + ' ' + str(res_sig[0,3, num]) + '\n')


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

def splitAndInfer(data,model):
    stripped_list = [s.strip("\n") for s in data]
    npData = np.zeros((4,2700))
    idx = 0
    for line in stripped_list:
        numlist = line.split(" ")
        npData[:,idx] = numlist
        idx += 1

    a1 = infer(npData[:,0:900],model)

    a2 = infer(npData[:,900:1800],model)
    a3 = infer(npData[:,1800:2700],model)
    a = (a1+a2+a3)/3
    # plot_singal_all(a[0,:,:])
    return a ,a1

def splitSig(data):
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

loss_func = nn.SmoothL1Loss()
kl_func = KL_Loss()
kl_func = kl_func.to(device)

if __name__ == '__main__':
    print("loading matlab environment")
    eng = matlab.engine.start_matlab()
    print("matlab environment done")
    print("loading model...")
    model_path = path + "best.pth"
    if _Net == N[0]:
        # from AllSnrModel.BigData.DRSNMoreLayer.ppNet_DRSN_MoreLayer_1d import *
        # sonicNet = NestedUNet_DRSN1d_MoreLayer(num_classes=1, input_channels=1).to(device)
        pass
    elif _Net == N[1]:
        from AllSnrModel.BigData.MoreLayerTCNMSE.DRSN_MoreLayer_tcn import *
        sonicNet = NestedUNet_DRSN1d_MoreLayer(num_classes=1, input_channels=1).to(device)
    elif _Net == N[2]:
        from trash.MidData.DRSNMoreLayerMSE.ppNet_DRSN_MoreLayer_1d import *
        sonicNet = NestedUNet_DRSN1d_MoreLayer(num_classes=1, input_channels=1).to(device)
    elif _Net == N[3]:
        from trash.MidData.LayerMoreTCNMSE.ppNet_DRSN_1d_TCN import *
        sonicNet = NestedUNet_DRSN1d_TCN(num_classes=1, input_channels=1).to(device)
    elif _Net == N[4]:
        from AllSnrModel.BigData.MoreLayerTCNMSE_SmoothL1.DRSN_MoreLayer_tcn import *
        sonicNet = NestedUNet_DRSN1d_MoreLayer(num_classes=1, input_channels=1).to(device)
    else:
        sonicNet = None

    sonicNet.load_state_dict(torch.load(model_path))
    sonicNet.eval()
    model = sonicNet.to(device)
    print("model loading done")
    print("============================")
    while snr == 10:
        resPath = f"Result_tmp/sir_{snr}/"
        print("----------------------")
        print("snr:",snr)
        print("----------------------")
        if not os.path.exists(resPath):
            os.makedirs(resPath)
        clearTXT(resPath)
        fileCount = 0
        print("============================")
        txtCount = 100
        for count in range(txtCount):
            print("====================")
            print("count:",count)
            labelpath = f"ValidateSIR/sir_{snr}/label/label_{count}.txt"
            datapath = f"ValidateSIR/sir_{snr}/data/data_{count}.txt"
            labsigpath = f"ValidateSIR/sir_{snr}/labelSig/data_{count}.txt"

            # labelpath = f"ValidateSIR/label_0.txt"
            # datapath = f"ValidateSIR/data_0.txt"
            # labsigpath = f"ValidateSIR/labsig_0.txt"

            s = time.time()
            with open(labelpath,'r') as f:
                labellines = f.readlines()
            with open(datapath,'r') as ff:
                datalines = ff.readlines()
            with open(labsigpath,'r') as fff:
                labsiglines = fff.readlines()

            for i in range(groupNum):
                data = datalines[i*2700:(i+1)*2700]
                labsig = labsiglines[i*2700:(i+1)*2700]
                label = labellines[i]
                labels = splitLabel(label)
                labsigs = splitSig(labsig)
                output,a1 = splitAndInfer(data,model)
                writefile_tensor(output)
                # plot_compare(a1,labsigs,data)

                print(" ")
                print(f"realTheta:{labels[0]}  realFai:{labels[1]}")
                eng.pyMatlibMusicNoDown(float(labels[0]), float(labels[1]),path,resPath)
                e = time.time()
                print("time:",e-s)

        calError(resPath)
        snr -= 1
