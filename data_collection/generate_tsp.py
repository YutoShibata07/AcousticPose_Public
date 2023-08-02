import numpy as np


def normal_tsp(n, gain=50, repeat=1):
    N = 2 ** n
    m = N // 4

    A = 50
    L = N // 2 - m
    k = np.arange(0, N)

    tsp_freq = np.zeros(N, dtype=np.complex128)
    tsp_exp = np.exp(-1j * 4 * m * np.pi * (k / N) ** 2)

    tsp_freq[0 : N // 2 + 1] = tsp_exp[0 : N // 2 + 1]
    tsp_freq[N // 2 + 1 : N + 1] = np.conj(tsp_exp[1 : N // 2][::-1])

    tsp_inv_freq = 1 / tsp_freq

    tsp = np.real(np.fft.ifft(tsp_freq))
    tsp = gain * np.roll(tsp, L)

    tsp_repeat = np.r_[np.tile(tsp, repeat), np.zeros(N)]

    tsp_inv = np.real(np.fft.ifft(tsp_inv_freq))
    tsp_inv = gain * np.roll(tsp_inv, -L)

    tsp_inv_repeat = np.r_[np.tile(tsp_inv, repeat), np.zeros(N)]

    return tsp_repeat, tsp_inv


def pink_tsp(n, gain=50, repeat=1):

    N = 2 ** n
    m = N // 4

    L = N // 2 - m
    k = np.arange(1, N)

    a = 4 * m * np.pi / (N * np.log(N / 2))

    tsp_freq = np.zeros(N, dtype=np.complex128)
    tsp_exp = np.exp(1.0j * a * k * np.log(k)) / np.sqrt(k)

    tsp_freq[0] = 1
    tsp_freq[1 : N // 2 + 1] = tsp_exp[1 : N // 2 + 1]
    tsp_freq[N // 2 + 1 : N + 1] = np.conj(tsp_exp[1 : N // 2][::-1])

    tsp_inv_freq = 1 / tsp_freq

    tsp = gain * np.real(np.fft.ifft(tsp_freq))[::-1]
    tsp = gain * np.roll(tsp, L)
    print(len(tsp))
    tsp_repeat = np.r_[np.tile(tsp, repeat), np.zeros(N)]
    print(len(tsp_repeat))
    tsp_inv = np.real(np.fft.ifft(tsp_inv_freq))[::-1]
    tsp_inv = gain * np.roll(tsp_inv, L)

    return tsp_repeat, tsp_inv


if __name__ == "__main__":
    import soundfile as sf
    import os

    output_dir = "tsp"
    n = 12

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    normal_tsp, normal_tsp_inv = normal_tsp(n, repeat=3000)
    pink_tsp, pink_tsp_inv = pink_tsp(n, repeat=3000)

    sf.write(os.path.join(output_dir, "normal_tsp_%s.wav" % n), normal_tsp, 48000)
    sf.write(os.path.join(output_dir, "pink_tsp_%s.wav" % n), pink_tsp, 48000)
    sf.write(
        os.path.join(output_dir, "normal_tsp_inv_%s.wav" % n), normal_tsp_inv, 48000
    )
    sf.write(os.path.join(output_dir, "pink_tsp_inv_%s.wav" % n), pink_tsp_inv, 48000)
