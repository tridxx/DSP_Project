# 所使用到的库函数
import numpy as np


# from scipy.fft import fft


# 想法：自己写一个spec_x=myfft(x)，还没写[5]DCT和h_mel，别的写完了
def mfcc_cal(spec_x, num_filter, n):  # s是Mel滤波器能量,n是阶数，num_filter是Mel滤波器数量,求某一帧s的DCT得到的mcff系数
    res = 0
    for _i in range(num_filter):
        res = res + np.log(spec_x[_i]) * np.cos(np.pi * n * (2 * num_filter - 1) / (2 * num_filter))
    return np.sqrt(2 / num_filter) * res


def energy_cal(spec_x, num_filter, _h_mel):  # spec_x是一帧的信号的频谱
    spec_energy = np.power(np.real(spec_x), 2) + np.power(np.imag(spec_x), 2)
    spec_mel = np.zeros(num_filter)
    for m in range(num_filter):
        spec_mel[m] = sum(np.multiply(spec_energy, _h_mel[m]))  # 相乘后求和
    return spec_mel


# 递归FFT，利用分治思想的dft
def fft_recurrence(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    x_even = fft_recurrence(x[0::2])
    x_odd = fft_recurrence(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(n) / n)

    return np.concatenate([x_even + factor[:int(n / 2)] * x_odd,
                           x_even + factor[int(n / 2):] * x_odd])


def main():
    t1 = np.linspace(0, 5 * np.pi, 200)  # 时间坐标
    x1 = np.sin(2 * np.pi * t1)  # 正弦函数
    h_mel = [0]
    # 输入x,然后进行分帧，分成x[i]
    num_frame = 1
    s_x = np.zeros(num_frame)
    num_melfilter = 10
    for i in range(len(x1)):
        s_x[i] = fft_recurrence(x1[i])  # 求fft变换

    s1_x = np.zeros((num_frame, num_melfilter))
    mfcc_x = np.zeros(num_frame)

    for i in range(num_frame):
        s1_x[i] = energy_cal(s_x[i], num_melfilter, h_mel)
        mfcc_x[i] = mfcc_cal(s1_x[i], num_melfilter, 12)

    print(mfcc_x)  # 输出


if __name__ == '__main__':
    main()
