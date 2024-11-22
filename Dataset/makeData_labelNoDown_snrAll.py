import time
from utils import *

#####################################################
snrlist = [0,40]  # SNR range you want to generate
rvbNum = 4  # multipath interference number
total_group_num = 1200  # Group number in a txt file(data is stored in txt format)

cycleNum = 3  # The number of cycle in a data frame, when set 0.1s as a cycle
squareNum = 1  # The number of square wave cycles within 0.1s, thereby controlling the period of the square wave
#####################################################

total_filecount = 100  # total file number
all_num = total_filecount * total_group_num
num_per_file = all_num / total_filecount
if_square = True  #
if_cycle = True
if_noise = True  #
if_InitPhaseAndDist = True
sq_phase = 0

soundF = 40000  # signal frequency
DsoundF = abs(soundF - np.floor(soundF / 3000 + 0.5) * 3000)  # frequency after bandpass sample
degrade = math.pi / 180
r = 0.002  # distance from each microphone to origin
l = 340 / soundF
fs = 1200e3
t = np.arange(0, 0.1, 1 / fs)  # 时间

t = t.reshape(1, -1)
# print(t.shape)
Angle = np.zeros((2, 3))
RAngle = np.zeros((2, 3))
maxDist = 16.9

fainum = 360
thenum = 90

s = time.time()
if __name__ == '__main__':
    print("==================================================")
    print("There are {} txt files in total. Each txt file contains {} sets of data. There are {} sets of data in total.".format(
        total_filecount, total_group_num, total_filecount * total_group_num))
    print("There are {} fai, {} theta, and a total of {} angle combinations. On average, each combination has {} data".format(
        fainum, thenum, fainum * thenum, (total_filecount * total_group_num) / (fainum * thenum)))
    print("==================================================")
    fileCount = 0
    failist = gen_fai(fainum)
    thelist = gen_the(thenum)
    for count in range(all_num):
        if count % num_per_file == 0 and count != 0:
            e = time.time()
            print("==================")
            print("fileCount:", fileCount)
            print("one file time:", e - s)
            s = time.time()
            fileCount += 1
        """
        realtheta is Pitch Angle，realfai is Horizontal Angle
        """
        # Generate a random distance, the range is 1-5m
        thetaNum = int(random.random() * thenum)
        faiNum = int(random.random() * fainum)
        theta = thelist[thetaNum]
        fai = failist[faiNum]

        Xtheta = theta * degrade
        Xfai = fai * degrade
        Angle[0, 0] = Xtheta  # 0,0 is Pitch Angle
        Angle[1, 0] = Xfai  # 1,0 is Horizontal Angle
        realTheta = Angle[0, 0] / degrade
        realFai = Angle[1, 0] / degrade

        if if_InitPhaseAndDist:
            Init_phase = random.uniform(0, 2 * math.pi)
            distance = random.random() * 4 + 1
        else:
            Init_phase = 0
            distance = 0

        # The initial phase can be directly limited to [0,T], T=1/f = 1/40khz
        A = generate_A(r, Angle)
        A = np.asarray(A)
        A = A.reshape(-1, 1)

        snr = np.random.randint(snrlist[0], snrlist[1] + 1)
        signalR = 100*generate_OriginSignal(A, Init_phase, distance,snr)
        labelSig = 100*generate_LabelSignal(A,Init_phase,distance,cycleNum)
        signalR = signalR

        if if_cycle:
            cyclefinalSig = np.zeros((cycleNum, 4, 300))
            for kk in range(cycleNum):
                final_signal = signalR
                # generate multipath
                reverberate_num = int(random.random() * (rvbNum + 1))
                for i in range(reverberate_num):
                    R_S = generate_reverberation(signalR, thelist, failist,snr)
                    final_signal = final_signal + R_S
                final_signal3000 = final_signal[:, ::400]
                cyclefinalSig[kk, :, :] = final_signal3000

        else:
            cyclefinalSig = np.zeros((4, 300))
            final_signal = signalR
            reverberate_num = int(random.random() * (rvbNum + 1))
            for i in range(reverberate_num):
                R_S = generate_reverberation(signalR, thelist, failist,snr)
                final_signal = final_signal + R_S

            final_signal3000 = final_signal[:, ::400]
            cyclefinalSig[:, :] = final_signal3000

        writefile_allSig(cyclefinalSig,labelSig,fileCount,count)
