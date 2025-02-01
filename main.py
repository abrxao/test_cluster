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
    print("\nConfigurando GPU...\n")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Avoid warnings from TensorFlow
    tf.get_logger().setLevel("ERROR")
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print("__________________")
        print("\n GPU Reconhecida ")
        print("__________________")
        gpu_num = 0  # Index of the GPU to be used
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
        try:
            # tf.config.set_visible_devices([], 'GPU')
            tf.config.set_visible_devices(gpus[gpu_num], "GPU")
            tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
        except RuntimeError as e:
            print(e)


def main():
    configTF()
    # consts
    fc = 3.5e9
    subcarrier_spacing = 15e3
    # num of subcarriers
    fft_size = 1024
    frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing) + fc
    # Angle of Arrival em graus
    AoA = 10

    tdl_rural = TDL(
        model="D",
        delay_spread=100e-9,  # microseconds
        carrier_frequency=fc,
        los_angle_of_arrival=AoA,
        min_speed=2,
        max_speed=20,
    )

    setNTNParamsTDL(tdl_rural, "A", "rural", "s_band")
    # Generate CIR
    cir = tdl_rural(
        batch_size=1, num_time_steps=1000, sampling_frequency=subcarrier_spacing
    )
    # Generate OFDM channel from CIR
    ofdm_channel = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    h_freq = tf.squeeze(ofdm_channel)
    h_freq = np.transpose(h_freq, (1, 0))
    ##############################################################3

    tdl_suburban = TDL(
        model="D",
        delay_spread=100e-9,  # microseconds
        carrier_frequency=fc,
        los_angle_of_arrival=AoA,
        min_speed=2,
        max_speed=20,
    )

    setNTNParamsTDL(tdl_suburban, "D", "suburban", "s_band")
    # Generate CIR
    cir = tdl_suburban(
        batch_size=1, num_time_steps=1000, sampling_frequency=subcarrier_spacing
    )
    # Generate OFDM channel from CIR
    ofdm_channel_s = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
    h_freq_s = tf.squeeze(ofdm_channel_s)
    h_freq_s = np.transpose(h_freq_s, (1, 0))
    # Visualize results
    ######## Plotting #########
    ts_1 = 0
    ts_2 = 99
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, linearToDB(np.abs(h_freq_s))[:, ts_1], "--", color="blue")
    plt.plot(
        frequencies,
        linearToDB(np.abs(h_freq_s))[:, ts_2],
        "-",
        color="blue",
        alpha=0.5,
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Channel frequency response (dB)")
    plt.grid(alpha=0.5)
    plt.legend([f"Suburban TS:{ts_1}", f"Suburban TS:{ts_2}"])

    ###################################################
    plt.subplot(1, 2, 2)
    plt.plot(frequencies, linearToDB(np.abs(h_freq))[:, ts_1], "--", color="orange")
    plt.plot(
        frequencies,
        linearToDB(np.abs(h_freq))[:, ts_2],
        "-",
        color="orange",
        alpha=0.5,
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Channel frequency response (dB)")
    plt.grid(alpha=0.5)
    plt.legend([f"TDL-A NTN TS:{ts_1}", f"TDL-A NTN TS:{ts_2}"])
    plt.show()


if __name__ == "__main__":
    main()
