# 所使用到的库函数
import numpy as np
from scipy.fft import fft

# 想法：自己写一个spec_x=myfft(x)，还没写[5]DCT和h_mel，别的写完了

t = np.linspace(0, 5 * np.pi, 200)  # 时间坐标
x = np.sin(2 * np.pi * t)  # 正弦函数

spec_x = fft(x)
spec_energy = np.power(np.real(spec_x), 2) + np.power(np.imag(spec_x), 2)
num_filter = 10
spec_mel = np.zeros(num_filter)
h_mel = np.zeros(num_filter)  # Mel滤波器的频率响应
one = np.ones(len(spec_energy))
print(spec_mel)
for m in range(len(spec_energy)):
    spec_mel[m] = np.multiply(spec_energy, h_mel[m]) * one  # 相乘后求和
