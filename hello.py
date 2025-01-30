import os  # Configure which GPU
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from lib.sionna_ext.setNTNParamsTDL import setNTNParamsTDL
from lib.utils import linearToDB
from sionna.channel.tr38901 import TDL
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)


def configTF():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Avoid warnings from TensorFlow
    tf.get_logger().setLevel("ERROR")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu_num = 0  # Index of the GPU to be used
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
        try:
            # tf.config.set_visible_devices([], 'GPU')
            tf.config.set_visible_devices(gpus[gpu_num], "GPU")
            tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
        except RuntimeError as e:
            print(e)


def main():
    # configTF()
    # consts
    fc = 3.5e9
    subcarrier_spacing = 15e3
    # num of subcarriers
    fft_size = 1024
    # num of symbols
    num_symbols = 1
    # bandwidth
    bandwidth = subcarrier_spacing * fft_size
    frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing) + fc

    tdl = TDL(
        model="D",
        delay_spread=100e-9,  # microseconds
        carrier_frequency=fc,
        los_angle_of_arrival=np.radians(50),
        min_speed=2,
        max_speed=20,
    )
    print("\n\n")
    print(
        tdl._los,
        "\n",
        tdl._scale_delays,
        "\n",
        tdl._delays,
        "\n",
        tdl._mean_powers,
        "\n",
        tdl._num_clusters,
    )

    # Generate CIR from tdl_origin
    cir = tdl(batch_size=1, num_time_steps=1000, sampling_frequency=subcarrier_spacing)
    # Generate OFDM channel from CIR
    ofdm_channel = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    h_freq_origin = tf.squeeze(ofdm_channel)
    h_freq_origin = np.transpose(h_freq_origin, (1, 0))
    setNTNParamsTDL(tdl, "A")
    print("\n\n")

    print(
        tdl._los,
        "\n",
        tdl._scale_delays,
        "\n",
        tdl._delays,
        "\n",
        tdl._mean_powers,
        "\n",
        tdl._num_clusters,
    )
    # Generate CIR
    cir = tdl(batch_size=1, num_time_steps=1000, sampling_frequency=subcarrier_spacing)
    # Generate OFDM channel from CIR
    ofdm_channel = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    h_freq = tf.squeeze(ofdm_channel)
    h_freq = np.transpose(h_freq, (1, 0))

    ts_1 = 0
    ts_2 = 99
    # Visualize results
    ######## Plotting #########
    """ plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, linearToDB(np.abs(h_freq))[:, ts_1], "--", color="blue")
    plt.plot(
        frequencies,
        linearToDB(np.abs(h_freq))[:, ts_2],
        "-",
        color="blue",
        alpha=0.5,
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Channel frequency response (dB)")
    plt.grid(alpha=0.5)
    plt.legend([f"TDL-A NTN TS:{ts_1}", f"TDL-A NTN TS:{ts_2}"])
    ######## Plotting Imaginary Part #########
    plt.subplot(1, 2, 2)
    plt.plot(
        frequencies,
        linearToDB(np.abs(h_freq_origin))[:, ts_1],
        "--",
        color="orange",
    )
    plt.plot(
        frequencies,
        linearToDB(np.abs(h_freq_origin))[:, ts_2],
        "-",
        color="orange",
        alpha=0.5,
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Channel frequency response (dB)")
    plt.grid(alpha=0.5)
    plt.legend([f"TDL-A  TS:{ts_1}", f"TDL-A TS:{ts_2}"])
    plt.show() """


if __name__ == "__main__":
    main()
