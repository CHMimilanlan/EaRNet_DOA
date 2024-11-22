import matplotlib.pyplot as plt
import pylab as mpl
import numpy.fft as fft
import math
import numpy as np
import random
from scipy import signal


def gen_fai(fai_num):  # 360
    failist = []
    for fai in range(fai_num):
        failist.append(fai)

    return failist


def gen_the(the_num):  # 180
    thetalist = []
    for the in range(the_num):
        thetalist.append(the)

    return thetalist


def rounding(array):
    array = array + 0.5
    array_int = array.astype(int)
    return array_int


def generate_A(r, Angle):
    A = [r * math.cos(Angle[0, 0]) * math.cos(Angle[1, 0]) / 340,
         (-r) * math.cos(Angle[0, 0]) * math.cos(Angle[1, 0]) / 340,
         r * math.sin(Angle[0, 0]) * math.cos(Angle[1, 0]) / 340,
         (-r) * math.sin(Angle[0, 0]) * math.cos(Angle[1, 0]) / 340]

    return A


mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def plot_data(signal3000, R_signal3000_1, R_signal3000_2, fianl_signal):
    xp = np.linspace(0, 100, 300)
    # yp1 = signal3000[0:100]
    plt.subplot(2, 2, 1)
    plt.plot(xp, signal3000[0, 0:300], 'bo-')
    plt.title("原数据")

    plt.subplot(2, 2, 2)
    plt.plot(xp, R_signal3000_1[0, 0:300], 'rx-')
    plt.title("混响1")

    plt.subplot(2, 2, 3)
    plt.plot(xp, R_signal3000_2[0, 0:300], 'rx-')
    plt.title("混响2")

    plt.subplot(2, 2, 4)
    plt.plot(xp, fianl_signal[0, 0:300], 'g')
    plt.title("最终数据")

    plt.show()


def plot_singal(data):
    length = len(data)
    xp1 = np.linspace(1, length, length)
    yp1 = data
    plt.plot(xp1, yp1, 'rx-')
    plt.show()


def plot_singal_all(data):
    x, y = data.shape
    for i in range(x):
        length = y
        xp1 = np.linspace(1, length, length)
        yp1 = data[i, :]
        plt.subplot(2, 2, i + 1)
        plt.plot(xp1, yp1, color='blue')
        plt.title("第{}通道".format(i + 1))

    plt.show()


def plot_compare(resSig,labelSig,OriginSig):

    x, y = resSig.shape
    singleLabel = np.zeros((1,900))
    singleLabel[0,0:300] = labelSig[0,0,:]
    singleLabel[0,300:600] = labelSig[1,0,:]
    singleLabel[0,600:900] = labelSig[2,0,:]
    singleOrigin = np.zeros((1,900))
    singleOrigin[0,0:300] = OriginSig[0,0,0,:]
    singleOrigin[0,300:600] = OriginSig[0,1,0,:]
    singleOrigin[0,600:900] = OriginSig[0,2,0,:]

    diff = resSig-singleLabel

    length = y
    xp1 = np.linspace(1, length, length)
    yp1 = resSig[0,:]
    yp2 = singleLabel[0,:]
    yp3 = singleOrigin[0,:]
    yp4 = diff[0,:]

    plt.subplot(4,1,1)
    plt.plot(xp1, yp3, color='blue')
    plt.title("原始信号")

    plt.subplot(4,1,2)
    plt.plot(xp1, yp1, color='blue')
    plt.title("推理结果图")

    plt.subplot(4,1,3)
    plt.plot(xp1, yp2, color='blue')
    plt.title("标签图")

    plt.subplot(4,1,4)
    plt.plot(xp1, yp4, color='blue')
    plt.title("误差图")
    plt.show()


def writediff_tensor(diffSig):
    x, y = diffSig.shape  # 4*300
    with open("diff.txt", 'w') as f:
        for num in range(y):
            f.write(str(diffSig[ 0, num]) + ' ' + str(diffSig[ 1, num]) + ' ' +
                    str(diffSig[ 2, num]) + ' ' + str(diffSig[ 3, num]) + '\n')


def plot_compare_NoPlus(resSig,labelSig,OriginSig):
    # labelsig 4*900
    x, y = resSig.shape
    singleLabel = np.zeros((4,900))
    for i in range(4):
        singleLabel[i,:] = labelSig[i,:]

    singleOrigin = np.zeros((4,900))
    for i in range(4):
        singleOrigin[i,0:300] = OriginSig[0,i,:]
        singleOrigin[i,300:600] = OriginSig[1,i,:]
        singleOrigin[i,600:900] = OriginSig[2,i,:]

    diff = resSig-singleLabel
    writediff_tensor(diff)

    print("SNR:", check_snr(singleLabel[1, :] - diff[1,:], diff[1,:]))

    length = y
    xp1 = np.linspace(1, length, length)
    yp1 = resSig[1,:]
    yp2 = singleLabel[1,:]
    yp3 = singleOrigin[1,:]
    yp4 = diff[1,:]

    plt.subplot(4,1,1)
    plt.plot(xp1, yp3, color='blue')
    plt.title("原始信号")

    plt.subplot(4,1,2)
    plt.plot(xp1, yp1, color='blue')
    plt.title("推理结果图")

    plt.subplot(4,1,3)
    plt.plot(xp1, yp2, color='blue')
    plt.title("标签图")

    plt.subplot(4,1,4)
    plt.plot(xp1, yp4, color='blue')
    plt.title("误差图")
    plt.show()


def plot_compare_fftDenoise(resSig,labelSig,fftDenoise):

    x, y = resSig.shape
    singleLabel = np.zeros((4,900))
    for i in range(4):
        singleLabel[i,0:300] = labelSig[0,i,:]
        singleLabel[i,300:600] = labelSig[1,i,:]
        singleLabel[i,600:900] = labelSig[2,i,:]

    singleOrigin = np.zeros((4,900))
    for i in range(4):
        singleOrigin[i,:] = fftDenoise[i,:]

    diff = singleOrigin-singleLabel
    print("range of diff loss",np.max(diff[1,:])-np.min(diff[1,:]))
    writediff_tensor(diff)

    print("SNR:", check_snr(singleLabel[1, :] - diff[1,:], diff[1,:]))

    length = y
    xp1 = np.linspace(1, length, length)
    yp1 = resSig[1,:]
    yp2 = singleLabel[1,:]
    yp3 = singleOrigin[1,:]
    yp4 = diff[1,:]

    plt.subplot(4,1,1)
    plt.plot(xp1, yp3, color='blue')
    plt.title("fft去噪")

    plt.subplot(4,1,2)
    plt.plot(xp1, yp1, color='blue')
    plt.title("推理结果图")

    plt.subplot(4,1,3)
    plt.plot(xp1, yp2, color='blue')
    plt.title("标签图")

    plt.subplot(4,1,4)
    plt.plot(xp1, yp4, color='blue')
    plt.title("误差图")
    plt.show()


def plot_compare_TensorResSig(resSig,labelSig,OriginSig,channel):
    # resSig -> 3*4*300
    # resSig = resSig.squeeze()
    singleRes = np.zeros((4,900))
    for i in range(4):
        singleRes[i,0:300] = resSig[0,0,i,:]
        singleRes[i,300:600] = resSig[0,1,i,:]
        singleRes[i,600:900] = resSig[0,2,i,:]

    singleLabel = np.zeros((4,900))
    for i in range(4):
        singleLabel[i,0:300] = labelSig[0,i,:]
        singleLabel[i,300:600] = labelSig[1,i,:]
        singleLabel[i,600:900] = labelSig[2,i,:]

    singleOrigin = np.zeros((4,900))
    for i in range(4):
        singleOrigin[i,0:300] = OriginSig[0,i,:]
        singleOrigin[i,300:600] = OriginSig[1,i,:]
        singleOrigin[i,600:900] = OriginSig[2,i,:]

    diff = singleRes-singleLabel
    writediff_tensor(diff)

    print("SNR:", check_snr(singleLabel[channel, :] - diff[channel,:], diff[channel,:]))

    length = 900
    xp1 = np.linspace(1, length, length)
    yp1 = singleRes[channel,:]
    yp2 = singleLabel[channel,:]
    yp3 = singleOrigin[channel,:]
    yp4 = diff[channel,:]

    plt.subplot(4,1,1)
    plt.plot(xp1, yp3, color='blue')
    plt.title("原始信号")

    plt.subplot(4,1,2)
    plt.plot(xp1, yp1, color='blue')
    plt.title("推理结果图")

    plt.subplot(4,1,3)
    plt.plot(xp1, yp2, color='blue')
    plt.title("标签图")

    plt.subplot(4,1,4)
    plt.plot(xp1, yp4, color='blue')
    plt.title("误差图")
    plt.show()


def gen_square_rvb(reverberation):
    # 300 - 400 | 120000 - 160000  | 50 - 20000
    new_rvb = np.zeros((4, 120000))
    pos = int(random.random() * 180000)
    # print("pos:",pos)
    if pos < 60000:
        # print("rid former")
        endp = pos
        new_rvb[:, :endp] = reverberation[:, :endp]

    else:
        startp = pos - 60000
        # print("startp:", startp)
        # print("do not rid")
        new_rvb[:, startp:startp + 60000] = reverberation[:, startp:startp + 60000]

    return new_rvb



def checkfftAndSig(sig):
    channelsig = sig[0,:]
    complex_array = fft.fft(channelsig)  # 快速傅里叶变换，返回结果为1000个复数，复数的模为幅值；复数的角度为相频
    complex_abs = abs(complex_array)  # 频谱
    xp1 = np.linspace(1, 300, 300)
    yp1 = complex_abs
    plt.subplot(1,2,1)
    plt.plot(xp1, yp1, color="blue")

    xp2 = np.linspace(1, 300, 300)
    yp2 = channelsig
    plt.subplot(1,2,2)
    plt.plot(xp2, yp2, color="pink")
    plt.show()








def gen_stepSig(Sinsig):
    """
    传入sin函数，生成阶跃函数，该函数由sin函数构成，可以对方波的占空比进行设置
    :return:方波
    """
    sq_sig = np.where(Sinsig >= 0, 1, 0)
    return sq_sig


def check_snr(signal, noise):
    """
    :param signal: 原始信号
    :param noise: 生成的高斯噪声
    :return: 返回两者的信噪比
    """
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  # 0.5722037
    noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))  # 0.90688
    SNR = 10 * np.log10(signal_power / noise_power)
    return SNR


def gen_gaussian_noise(signal, SNR):
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise = np.random.randn(*signal.shape)  # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  # 信号能量
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    return noise


def phase_convolution(signal, component_num, Soundfreq, Samplefreq):
    t = np.arange(0, 0.1, 1 / Samplefreq)
    spl = np.linspace(0, 1, component_num + 1)
    spl = spl[0:-1]
    spl = np.asarray(spl)
    split_phase = (2 * math.pi) * spl
    length = len(split_phase)
    reslist = np.zeros((4, 100))

    for chan in range(4):
        for i in range(length):
            # a = i / length
            sigS = np.sin(2 * math.pi * Soundfreq * t + split_phase[i])
            sigS = sigS[::400]
            sigS.reshape(300, 1)
            res = signal[chan, :] @ sigS
            reslist[chan, i] = res

    return reslist




def writefile_TEST(res_sig):
    x, y = res_sig.shape  # 4*300
    with open("testdata.txt", 'w') as f:
        for num in range(y):
            f.write(str(res_sig[0, num]) + ' ' + str(res_sig[1, num]) + ' ' +
                    str(res_sig[2, num]) + ' ' + str(res_sig[3, num]) + '\n')


def triangleWave_v1(sig):
    # y=kx+b,然后结果和原始信号相乘
    pass
    triangle = np.zeros((4,120000))
    t_linear = np.linspace(1,0,60000)
    triangle[:,:60000] = t_linear
    resSig = triangle*sig
    return resSig


def check_power_spectrum(sig):
    # 计算信号的功率谱密度
    freq, psd = signal.welch(sig, 3000, nperseg=256)
    # 绘制功率谱密度图
    plt.plot(freq, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.show()

def fftDenoise(signal):
    DenoiseSig = np.zeros((4,900))
    for i in range(4):
        dt = 1 / 3000
        sig = signal[i,:]
        n = sig.shape
        n = n[0]
        fhat = np.fft.fft(sig,n)
        PSD = fhat * np.conj(fhat) / n
        # print(np.argmax(PSD))

        freq = (1 / (dt * n)) * np.arange(n)
        L = np.arange(1,np.floor(n/2),dtype='int')

        indices = PSD > 100

        PSDclean = PSD * indices
        fhat = indices * fhat

        # plt.plot(freq[L], PSD[L], color='c', label='PSD')
        # plt.xlim(freq[L[0]], freq[L[-1]])
        # plt.show()
        # # #
        # plt.plot(freq[L], PSDclean[L], color='c', label='PSDclean')
        # plt.xlim(freq[L[0]], freq[L[-1]])
        # plt.show()

        ffilt = np.fft.ifft(fhat)
        DenoiseSig[i,:] = ffilt

    return DenoiseSig
