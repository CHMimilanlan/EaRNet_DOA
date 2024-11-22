import numpy as np
import matplotlib.pyplot as plt
import math

def calError(path):
    totaldata = []
    with open(path+"error.txt") as f:
        for data in f.readlines():
            onedata = data.strip("\n")
            onedata = float(onedata)
            totaldata.append(onedata)

    print(totaldata)
    totaldata.remove(np.max(totaldata))
    count = len(totaldata)
    means = np.mean(totaldata)

    mid = np.median(totaldata)
    var = np.var(totaldata)
    max = np.max(totaldata)
    std = np.std(totaldata)
    r_up = means + std/2
    r_down = means - std/2
    if r_down<0:
        r_down = 0

    print(f"means:{means},\nmid:{mid},\nvariance:{var}\nstd:{std}\nmax:{max}\nr_up:{r_up}\nr_down:{r_down}\n")

    with open(path+"CalError.txt","w") as f:
        f.write(f"means:{means},\nmid:{mid},\nvariance:{var}\nstd:{std}\nmax:{max}\nr_up:{r_up}\nr_down:{r_down}\n")

    xp1 = np.linspace(0,count,count)
    yp1 = totaldata
    plt.figure()
    plt.plot(xp1, yp1, linewidth=1, linestyle="solid", label="error")
    plt.savefig(path+"error.png")
    # plt.show()
    plt.close()

# calError("Result_Net_RMSE/snr_40/")


def calRMSE(path):
    totaldata = []
    with open(path+"error.txt") as f:
        for data in f.readlines():
            onedata = data.strip("\n")
            onedata = float(onedata)
            totaldata.append(onedata)

    # print(totaldata)
    totaldata.remove(np.max(totaldata))
    count = len(totaldata)
    rmse = math.sqrt(np.sum(np.power(totaldata,2))/count)
    print(rmse)

for i in range(21):
    if i == 6:
        continue
    calError(f"Result_SingleMusic/sir_{i}/")