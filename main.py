from lib.sionna_ext.TDL_NTN import TDL_NTN
from lib.delaySpreadNTNTDL import getScenarioParams
from lib.utils import genRandomDistances, calculatePathLoss

# from sionna.channel.tr38901 import TDL
from lib.configTensorFlow import configTensorflow
from sionna.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)
import matplotlib.pyplot as plt
from lib.utils import linearToDB
import numpy as np
import tensorflow as tf

# Getting Tensorflow configured for use GPU
configTensorflow(useGPU=False)


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
    AoA = 10
    # max speed for doppler shift
    maxS = 20
    # min speed for doppler shift
    minS = 2
    # Number of time steps
    n_ts = 10
    # Number of batch size
    n_bs = 1
    # Number of sattelite antenas
    n_tx_ant = 124
    # Number of users
    n_users = 1
    # Setting initial TDL model
    model = "A"

    # Getting delay spread
    sc_params = getScenarioParams(model, "dense_urban", fc, AoA)
    # TDL
    tdl_a = TDL_NTN(
        model=model,
        delay_spread=sc_params["delay_spread"],
        carrier_frequency=fc,
        los_angle_of_arrival=AoA,
        min_speed=minS,
        max_speed=maxS,
        num_tx_ant=n_tx_ant,
        num_rx_ant=n_users,
        ntn=True,
    )
    # Generate CIR
    a, tau = tdl_a(batch_size=n_bs, num_time_steps=n_ts, sampling_frequency=sc_spac)
    # Mean distance between users and sattelite
    mean_distances = 600e3
    # Distances variances
    var = (20e3) ** 2
    # Generate random distances
    distances = genRandomDistances(mean_distances, var, a.shape)

    """
        duvida: Na equação do path loss, ele diz que fc é em GHz, então eu preciso
        dividir a fc por 1e9?
        
        duvida: Enquanto o canal é gerado se faz necessário saber se o modelo TDL é los 
        ou nlos antes de cria-lo usando o sionna, mas o modelo TDL precisa de um
        delay spread, que é diferente se for los ou nlos.
        
        Podemos usar em uma simulação ka-band e s-band? Se sim, é preciso checar 
        diferença de parametros de path loss entre as bandas.
        
        Seria interessante calcular o path loss em escala Linear, pois os
        taps(a, tau) que o sionna retorna são em escala linear.
        
    """
    pl_linear = calculatePathLoss(fc, distances, sc_params, a.shape)
    a *= pl_linear
    # Generate OFDM channel from CIR
    ofdm_channel_a = cir_to_ofdm_channel(frequencies, a, tau)
    h_freq_a = tf.squeeze(ofdm_channel_a)
    # Combine antennas
    h_freq_a = tf.reduce_sum(h_freq_a, axis=0)

    h_freq_a = tf.transpose(h_freq_a, perm=[1, 0])
    ts_1 = 0
    ts_2 = 9
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
    plt.show()


if __name__ == "__main__":
    main()
