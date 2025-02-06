from lib.sionna_ext.setNTNParamsTDL import setNTNParamsToTDL
from sionna.channel.tr38901 import TDL
from lib.configTensorFlow import configGPUTensorflow
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)
import matplotlib.pyplot as plt
from lib.utils import linearToDB
import numpy as np

# Getting Tensorflow configured for use GPU
tf = configGPUTensorflow()


def main():
    # The values below were chosen for demonstration purposes only
    # Carrier frenquency
    fc = 3.5e9
    # Subcarrier spacing
    sc_spac = 15e3
    # Num of subcarriers
    fft_size = 1024
    # Array of all frequencies in scene
    frequencies = subcarrier_frequencies(fft_size, sc_spac) + fc
    # Reducing array to optimize memory
    frequencies = frequencies[6::12]
    # Angle of Arrival in degrees
    AoA = 50
    # max speed for doppler shift
    maxS = 20
    # min speed for doppler shift
    minS = 2
    # Number of time steps
    n_ts = 1000

    # Setting initial TDL
    tdl_a = TDL(
        model="A",
        # Setting a initial delay spread
        delay_spread=100e-9,  # microseconds
        carrier_frequency=fc,
        los_angle_of_arrival=AoA,
        min_speed=minS,
        max_speed=maxS,
    )

    setNTNParamsToTDL(tdl_a, "A", "dense_urban", "s_band")
    # Generate CIR
    cir_a = tdl_a(batch_size=1, num_time_steps=n_ts, sampling_frequency=sc_spac)
    # Generate OFDM channel from CIR
    ofdm_channel_a = cir_to_ofdm_channel(frequencies, *cir_a, normalize=True)
    h_freq_a = tf.squeeze(ofdm_channel_a)
    h_freq_a = tf.transpose(h_freq_a, perm=[1, 0])

    # Setting initial TDL
    tdl_d = TDL(
        model="B",
        # Setting a initial delay spread
        delay_spread=100e-9,  # microseconds
        carrier_frequency=fc,
        los_angle_of_arrival=AoA,
        min_speed=minS,
        max_speed=maxS,
    )

    setNTNParamsToTDL(tdl_d, "D", "dense_urban", "s_band")
    # Generate CIR
    cir_d = tdl_a(batch_size=1, num_time_steps=n_ts, sampling_frequency=sc_spac)
    # Generate OFDM channel from CIR
    ofdm_channel_d = cir_to_ofdm_channel(frequencies, *cir_d, normalize=True)
    h_freq_d = tf.squeeze(ofdm_channel_d)
    h_freq_d = tf.transpose(h_freq_d, perm=[1, 0])

    ts_1 = 0
    ts_2 = 100
    frequencies *= 1e-9
    plt.figure(figsize=(15, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(frequencies, linearToDB(np.abs(h_freq_a))[:, ts_1], "--", color="blue")
    ax1.plot(
        frequencies,
        linearToDB(np.abs(h_freq_a))[:, ts_2],
        "-",
        color="blue",
        alpha=0.5,
    )
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel(r"Channel frequency response (dB)")
    ax1.grid(alpha=0.5)
    ax1.legend([f"TDL-A NTN TS:{ts_1}", f"TDL-A NTN TS:{ts_2}"])
    ######## Plotting Imaginary Part #########
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(
        frequencies,
        linearToDB(np.abs(h_freq_d))[:, ts_1],
        "--",
        color="orange",
    )
    ax2.plot(
        frequencies,
        linearToDB(np.abs(h_freq_d))[:, ts_2],
        "-",
        color="orange",
        alpha=0.5,
    )
    ax2.grid(alpha=0.5)
    ax2.set_xlabel("Frequency (GHz)")
    ax2.legend([f"TDL-D  TS:{ts_1}", f"TDL-D TS:{ts_2}"])
    ax2.sharey(ax1)
    plt.show()


if __name__ == "__main__":
    main()
