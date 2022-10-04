# 所使用到的库函数
import numpy as np
from scipy.fft import fft


# 想法：自己写一个spec_x=myfft(x)，还没写[5]DCT和h_mel，别的写完了
def mcff(spec_x, num_filter, n):  # s是Mel滤波器能量,n是阶数，num_filter是Mel滤波器数量,求某一帧s的DCT得到的mcff系数
    res = 0
    for i in range(num_filter):
        res = res + np.log(spec_x[num_filter]) * np.cos(np.pi * n * (2 * num_filter - 1) / (2 * num_filter))
    return np.sqrt(2 / num_filter) * res


def energy_cal(spec_x, num_filter, h_mel):  # spec_x是一帧的信号的频谱
    spec_energy = np.power(np.real(spec_x), 2) + np.power(np.imag(spec_x), 2)
    spec_mel = np.zeros(num_filter)
    one = np.ones(len(spec_x))
    for m in range(num_filter):
        spec_mel[m] = np.multiply(spec_energy, h_mel[m]) * one  # 相乘后求和


t = np.linspace(0, 5 * np.pi, 200)  # 时间坐标
x = np.sin(2 * np.pi * t)  # 正弦函数
# 下面写的都是第i帧的，需要再修改
s = fft(x)
num = 10
