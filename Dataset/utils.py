import matplotlib.pyplot as plt
import pylab as mpl
import numpy.fft as fft
import math
import numpy as np
import random
from makeData_labelNoDown_snrAll import \
    if_noise,fainum,thenum,fs,squareNum,degrade,soundF,t,if_square,all_num,sq_phase,RAngle,r,maxDist
from scipy import signal

def Amp_Reduce(dist):
    if dist < maxDist:
        amp = 9 / dist - 0.4
        return amp, True
    else:
        return 0, False


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


mpl.rcParams['font.sans-serif'] = ['FangSong']  #
mpl.rcParams['axes.unicode_minus'] = False  #


def plot_data(signal3000, R_signal3000_1, R_signal3000_2, fianl_signal):
    xp = np.linspace(0, 100, 300)
    # yp1 = signal3000[0:100]
    plt.subplot(2, 2, 1)
    plt.plot(xp, signal3000[0, 0:300], 'bo-')
    plt.title("Origin")

    plt.subplot(2, 2, 2)
    plt.plot(xp, R_signal3000_1[0, 0:300], 'rx-')
    plt.title("multipath 1")

    plt.subplot(2, 2, 3)
    plt.plot(xp, R_signal3000_2[0, 0:300], 'rx-')
    plt.title("multipath 2")

    plt.subplot(2, 2, 4)
    plt.plot(xp, fianl_signal[0, 0:300], 'g')
    plt.title("final data")

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
        plt.title("channel {}".format(i + 1))

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
    plt.title("Origin signal")

    plt.subplot(4,1,2)
    plt.plot(xp1, yp1, color='blue')
    plt.title("Inference")

    plt.subplot(4,1,3)
    plt.plot(xp1, yp2, color='blue')
    plt.title("label")

    plt.subplot(4,1,4)
    plt.plot(xp1, yp4, color='blue')
    plt.title("error")
    plt.show()


def writediff_tensor(diffSig):
    x, y = diffSig.shape  # 4*300
    with open("diff.txt", 'w') as f:
        for num in range(y):
            f.write(str(diffSig[ 0, num]) + ' ' + str(diffSig[ 1, num]) + ' ' +
                    str(diffSig[ 2, num]) + ' ' + str(diffSig[ 3, num]) + '\n')


def plot_compare_NoPlus(resSig,labelSig,OriginSig):

    x, y = resSig.shape
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
    plt.title("origin")

    plt.subplot(4,1,2)
    plt.plot(xp1, yp1, color='blue')
    plt.title("inference")

    plt.subplot(4,1,3)
    plt.plot(xp1, yp2, color='blue')
    plt.title("label")

    plt.subplot(4,1,4)
    plt.plot(xp1, yp4, color='blue')
    plt.title("error")
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
    plt.title("fft denoise")

    plt.subplot(4,1,2)
    plt.plot(xp1, yp1, color='blue')
    plt.title("inference")

    plt.subplot(4,1,3)
    plt.plot(xp1, yp2, color='blue')
    plt.title("label")

    plt.subplot(4,1,4)
    plt.plot(xp1, yp4, color='blue')
    plt.title("error")
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
    plt.title("origin")

    plt.subplot(4,1,2)
    plt.plot(xp1, yp1, color='blue')
    plt.title("inference")

    plt.subplot(4,1,3)
    plt.plot(xp1, yp2, color='blue')
    plt.title("label")

    plt.subplot(4,1,4)
    plt.plot(xp1, yp4, color='blue')
    plt.title("error")
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

def delay_and_reduce_sig(signalR, delay_t, dist,snr):
    """
    Delay the main path signal according to time and attenuate according to distance
    :return:
    """
    amp, flag = Amp_Reduce(dist)
    if flag:
        sig = signalR * amp
        # print("sig[0,0]:{},sig[1,0]:{},sig[2,0]:{},sig[3,0]:{}".format(sig[0, 0], sig[1, 0], sig[2, 0], sig[3, 0]))
        delaySig = np.zeros(signalR.shape)
        for d in range(4):
            index = delay_t[d, 0]
            # print("dist:{},amp:{},index:{}".format(dist, amp, index))
            delaySig[d, index:index + 60001] = sig[d, 0:60001]
            # plot_singal(delaySig[d])

    else:
        delaySig = np.zeros(signalR.shape)

    if if_noise:
        for i in range(delaySig.shape[0]):
            noise = gen_gaussian_noise(delaySig[i, :], snr)
            delaySig[i, :] += noise

    return delaySig


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




def generate_reverberation(sigR, thelist, failist,snr):
    """
    The reverberation here is to randomly specify a sound source at a certain position in the space,
    and the distance from the position to the microphone is greater than the distance from the ultrasonic wave to the microphone.
    Since the main signal is a square wave, the generated reverberation signal is also a square wave with a maximum duty cycle of 50%.
    """
    dist = random.random() * 8 + 10
    RthetaNum = int(random.random() * thenum)
    RfaiNum = int(random.random() * fainum)
    Rtheta = thelist[RthetaNum]
    Rfai = failist[RfaiNum]
    RXtheta = Rtheta * degrade
    RXfai = Rfai * degrade
    RAngle[0, 0] = RXtheta
    RAngle[1, 0] = RXfai
    R_A = generate_A(r, RAngle)
    R_A = np.asarray(R_A)
    R_A = R_A.reshape(-1, 1)  # (4,1)
    delay = np.ones((4, 1))
    delay = delay * (dist / 340)
    delay += R_A
    delay_t = delay / (1 / fs)
    delay_t = rounding(delay_t)
    # 计算时间延时的时候要把R_A算上，但是计算幅度衰减的时候可以不加上，因为幅度衰减量之间的区别过小，可以忽略不计
    R_signal = delay_and_reduce_sig(sigR, delay_t, dist,snr)
    # print("R_signal:{}|{}|{}|{}".format(R_signal[0,delay_t[0,0]],R_signal[1,delay_t[1,0]],R_signal[2,delay_t[2,0]],R_signal[3,delay_t[3,0]]))
    # amp = Amp_Reduce(dist)
    # reverberation = amp * np.sin(2 * math.pi * soundF * (np.ones((4, 1)) @ t - R_A @ np.ones((1, t_y))) + Init_phase)
    # 注意这里，反射信号与主路径信号一致，且是主路径在时间上的偏移，这个偏移要先计算虚拟声音与原点的距离，然后在算上R_A
    # Init_phase 与主路径相同，R_A @ np.ones((1, len(t)))) 减去传播时间
    # R_signal = gen_square_rvb(reverberation)
    # 混响的方波与主路径相关
    return R_signal


def gen_stepSig(Sinsig):
    """
    Pass in the sin function to generate a step function, which is composed of the sin function and can set the duty cycle of the square wave
    :return:
    """
    sq_sig = np.where(Sinsig >= 0, 1, 0)
    return sq_sig


def gen_square(phase=sq_phase):
    # The square wave has a period of 0.1s, squareT, and the input function of the step function is the sin function with a period of 0.1s
    # It is known that the global variable is the number of square wave periods within 0.1s,
    squareT = 0.1 / squareNum
    w = 2 * math.pi / squareT  # 20pi
    tt = np.arange(0, 0.1, 1 / fs)  # 时间
    sig = np.sin(w * tt - phase)
    sqsig = gen_stepSig(sig)
    return sqsig
    # plot_singal(sqsig)


def generate_OriginSignal(A, Init_phase, distance,snr):
    t_x, t_y = t.shape
    # ss =np.ones((4, 1)) @ t - A @ np.ones((1, t_y))
    signalR = np.sin(2 * math.pi * soundF * (np.ones((4, 1)) @ t - distance / 340 - A @ np.ones((1, t_y))) + Init_phase)

    # A @ np.ones((1, len(t)))
    # Generate the original signal first, then add noise, and finally follow the square wave, setting the latter part to 0
    if if_noise:
        for i in range(signalR.shape[0]):
            noise = gen_gaussian_noise(signalR[i, :], snr)
            signalR[i, :] += noise
            # print("SNR:", check_snr(signalR[i,:]-noise, noise))

    if if_square:
        s = gen_square()
        signalR = signalR * s

    return signalR


def generate_LabelSignal(A, Init_phase, distance,cycleNum):
    t_x, t_y = t.shape
    labelsignalR = np.sin(2 * math.pi * soundF * (np.ones((4, 1)) @ t - distance / 340 - A @ np.ones((1, t_y))) + Init_phase)
    labelsignalR = labelsignalR[:,:900]
    # cyclefinalSig = np.zeros((cycleNum, 4, 300))
    # for kk in range(cycleNum):
    #     final_signal3000 = labelsignalR[:, 300*kk:300*(kk+1)]
    #     cyclefinalSig[kk, :, :] = final_signal3000
    # return cyclefinalSig
    return labelsignalR


def check_snr(signal, noise):
    """
    :param signal: origin
    :param noise: gaussian noise
    :return: snr
    """
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  # 0.5722037
    noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))  # 0.90688
    SNR = 10 * np.log10(signal_power / noise_power)
    return SNR


def gen_gaussian_noise(signal, SNR):

    noise = np.random.randn(*signal.shape)  # *signal.shape Get the size of the sample sequence
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  #
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


def writefile(res_sig, theta, fai, fileCount, count):
    """
    :param res_sig: 3*4*300
    :param fileCount: which file should be written
    :param count: the count of single res_sig data
    :return: None
    """
    x, y, z = res_sig.shape
    npf = z
    if count <= all_num * 0.9:
        with open("FineTuningUpSampleData/label/train/label_{}.txt".format(fileCount), 'a') as f:
            f.write(str(theta) + ' ' + str(fai) + ' ' + '\n')

        with open("FineTuningUpSampleData/data/train/data_{}.txt".format(fileCount), 'a') as f:
            for dim in range(x):
                for num in range(npf):
                    f.write(str(res_sig[dim,0, num]) + ' ' + str(res_sig[dim,1, num]) + ' ' +
                            str(res_sig[dim,2, num]) + ' ' + str(res_sig[dim,3, num]) + '\n')

    elif count > all_num * 0.9:
        with open("FineTuningUpSampleData/label/test/label_{}.txt".format(fileCount), 'a') as f:
            f.write(str(theta) + ' ' + str(fai) + ' ' + '\n')

        with open("FineTuningUpSampleData/data/test/data_{}.txt".format(fileCount), 'a') as f:
            for dim in range(x):
                for num in range(npf):
                    f.write(str(res_sig[dim,0, num]) + ' ' + str(res_sig[dim,1, num]) + ' ' +
                            str(res_sig[dim,2, num]) + ' ' + str(res_sig[dim,3, num]) + '\n')

def writefile_allSig(res_sig, label_sig, fileCount, count):
    """
    :param res_sig: 3*4*300
    :param fileCount: which file should be written
    :param count: the count of single res_sig data
    :return: None
    """
    x, y, z = res_sig.shape  # 3,4,300
    a,b = label_sig.shape  # 4,900
    npf = z
    if count <= all_num * 0.9:
        with open("UpSampleSnrAllData/label/train/label_{}.txt".format(fileCount), 'a') as f:
            for num in range(b):
                f.write(str(label_sig[0, num]) + ' ' + str(label_sig[1, num]) + ' ' +
                        str(label_sig[2, num]) + ' ' + str(label_sig[3, num]) + '\n')

        with open("UpSampleSnrAllData/data/train/data_{}.txt".format(fileCount), 'a') as f:
            for dim in range(x):
                for num in range(npf):
                    f.write(str(res_sig[dim,0, num]) + ' ' + str(res_sig[dim,1, num]) + ' ' +
                            str(res_sig[dim,2, num]) + ' ' + str(res_sig[dim,3, num]) + '\n')

    elif count > all_num * 0.9:
        with open("UpSampleSnrAllData/label/test/label_{}.txt".format(fileCount), 'a') as f:
            for num in range(b):
                f.write(str(label_sig[0, num]) + ' ' + str(label_sig[1, num]) + ' ' +
                        str(label_sig[2, num]) + ' ' + str(label_sig[3, num]) + '\n')

        with open("UpSampleSnrAllData/data/test/data_{}.txt".format(fileCount), 'a') as f:
            for dim in range(x):
                for num in range(npf):
                    f.write(str(res_sig[dim,0, num]) + ' ' + str(res_sig[dim,1, num]) + ' ' +
                            str(res_sig[dim,2, num]) + ' ' + str(res_sig[dim,3, num]) + '\n')


def writefile_TEST(res_sig):
    x, y = res_sig.shape  # 4*300
    with open("testdata.txt", 'w') as f:
        for num in range(y):
            f.write(str(res_sig[0, num]) + ' ' + str(res_sig[1, num]) + ' ' +
                    str(res_sig[2, num]) + ' ' + str(res_sig[3, num]) + '\n')


def triangleWave_v1(sig):
    # y=kx+b,The result is then multiplied by the original signal
    pass
    triangle = np.zeros((4,120000))
    t_linear = np.linspace(1,0,60000)
    triangle[:,:60000] = t_linear
    resSig = triangle*sig
    return resSig


def filter_bp(x,fs,wl,wh):  # fs = 1024
    fN = 3
    fc = fs/2
    w1c = wl/fc
    w2c = wh/fc
    b, a = signal.butter(fN,[w1c, w2c],'bandpass')
    x_filter = signal.filtfilt(b,a,x)
    return x_filter


def check_power_spectrum(sig):
    # Calculate the power spectral density of a signal
    freq, psd = signal.welch(sig, 3000, nperseg=256)
    # Plotting the power spectral density
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

        ffilt = np.fft.ifft(fhat)
        DenoiseSig[i,:] = ffilt

    return DenoiseSig


def writelabel_tensor(labelsig,path):
    x, y = labelsig.shape  # 4*300
    with open(path+"testlabel.txt", 'w') as f:
        for num in range(y):
            f.write(str(labelsig[0, num]) + ' ' + str(labelsig[1, num]) + ' ' +
                    str(labelsig[2, num]) + ' ' + str(labelsig[3, num]) + '\n')
