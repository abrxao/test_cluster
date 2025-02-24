from sionna_ext.TDL_NTN import TDL_NTN
from delaySpreadNTNTDL import getDelaySpread
import time
import sys

# from sionna.channel.tr38901 import TDL
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)
import numpy as np
import tensorflow as tf
from configTensorFlow import configTensorflow
import csv


def save_matrix_to_csv(matrix, filename):
    """Saves a 2D NumPy matrix to a CSV file.

    Args:
        matrix: The 2D NumPy matrix to save.
        filename: The name of the CSV file (e.g., "matrix.csv").
    """

    try:
        with open(
            filename, "w", newline="", encoding="utf-8"
        ) as csvfile:  # Use utf-8 encoding
            writer = csv.writer(csvfile)

            # Option 1: Save the matrix directly (each row as a CSV row)
            writer.writerows(matrix)  # Efficient for NumPy matrices

            # Option 2:  If your matrix is a NumPy array:
            # np.savetxt(csvfile, matrix, delimiter=",") # More concise for NumPy

    except Exception as e:
        print(f"Error saving matrix to CSV: {e}")


n_users = [10, 50, 100, 200, 500]
batch_size = [10, 20, 30, 40]


def testDevices(useGPU):
    configTensorflow(useGPU)

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
    n_ts = 15

    # Execution time of each pair of parameters
    execution_time_mtx = np.zeros((len(n_users), len(batch_size)))
    # Getting delay spread
    ds_ntn = getDelaySpread("los", "dense_urban", "s_band", AoA)
    # Num of monte carlo simulations
    n_mc = 100
    for _ in range(n_mc):
        for i_n, n_user in enumerate(n_users):
            for i_b, b_size in enumerate(batch_size):
                # Timer begin
                begin_t = time.perf_counter()
                tdl_a = TDL_NTN(
                    model="A",
                    delay_spread=ds_ntn,
                    carrier_frequency=fc,
                    los_angle_of_arrival=AoA,
                    min_speed=minS,
                    max_speed=maxS,
                    num_rx_ant=n_user,
                    ntn=True,
                )
                # Generate CIR
                cir_a = tdl_a(
                    batch_size=b_size, num_time_steps=n_ts, sampling_frequency=sc_spac
                )
                # Generate OFDM channel from CIR
                ofdm_channel_a = cir_to_ofdm_channel(
                    frequencies, *cir_a, normalize=True
                )
                h_freq_a = tf.squeeze(ofdm_channel_a)
                # Timer end
                end_t = time.perf_counter()
                execution_time_mtx[i_n, i_b] += end_t - begin_t
    simu_date = time.time()
    filename = "gpudata" if useGPU else "cpudata"
    save_matrix_to_csv(execution_time_mtx, f"{filename}{simu_date}.csv")


# Testing with GPU
testDevices(useGPU=True)
