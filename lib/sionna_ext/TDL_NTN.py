import json
from importlib_resources import files
import numpy as np
import tensorflow as tf
from . import models
from sionna.channel.tr38901 import TDL


class TDL_NTN(TDL):
    def __init__(
        self,
        model,
        delay_spread,
        carrier_frequency,
        num_sinusoids=20,
        los_angle_of_arrival=np.pi / 4.0,
        min_speed=0.0,
        max_speed=None,
        num_rx_ant=1,
        num_tx_ant=1,
        spatial_corr_mat=None,
        rx_corr_mat=None,
        tx_corr_mat=None,
        dtype=tf.complex64,
        ntn=False,  # Novo parâmetro adicionado
    ):
        self.ntn = ntn  # Armazena o novo parâmetro
        # Substituir a função _load_parameters se ntn for True
        if self.ntn:
            self._load_parameters = self._load_parameters_ntn
        super().__init__(
            model,
            delay_spread,
            carrier_frequency,
            num_sinusoids,
            los_angle_of_arrival,
            min_speed,
            max_speed,
            num_rx_ant,
            num_tx_ant,
            spatial_corr_mat,
            rx_corr_mat,
            tx_corr_mat,
            dtype,
        )

    def _load_parameters_ntn(self, fname):
        r"""Load parameters of a TDL model.

        The model parameters are stored as JSON files with the following keys:
        * los : boolean that indicates if the model is a LoS model
        * num_clusters : integer corresponding to the number of clusters (paths)
        * delays : List of path delays in ascending order normalized by the RMS
            delay spread
        * powers : List of path powers in dB scale

        For LoS models, the two first paths have zero delay, and are assumed
        to correspond to the specular and NLoS component, in this order.

        Input
        ------
        fname : str
            File from which to load the parameters.

        Output
        ------
        None
        """
        source = files(models).joinpath(fname)
        # pylint: disable=unspecified-encoding
        with open(source) as parameter_file:
            params = json.load(parameter_file)

        # LoS scenario ?
        self._los = bool(params["los"])

        # Scale the delays
        self._scale_delays = bool(params["scale_delays"])

        # Loading cluster delays and mean powers
        self._num_clusters = tf.constant(params["num_clusters"], tf.int32)

        # Retrieve power and delays
        delays = tf.constant(params["delays"], self._real_dtype)
        mean_powers = np.power(10.0, np.array(params["powers"]) / 10.0)
        mean_powers = tf.constant(mean_powers, self._dtype)

        if self._los:
            # The power of the specular component of the first path is stored
            # separately
            self._los_power = mean_powers[0]
            mean_powers = mean_powers[1:]
            # The first two paths have 0 delays as they correspond to the
            # specular and reflected components of the first path.
            # We need to keep only one.
            delays = delays[1:]

        # Normalize the PDP
        if self._los:
            norm_factor = tf.reduce_sum(mean_powers) + self._los_power
            self._los_power = self._los_power / norm_factor
            mean_powers = mean_powers / norm_factor
        else:
            norm_factor = tf.reduce_sum(mean_powers)
            mean_powers = mean_powers / norm_factor

        self._delays = delays
        self._mean_powers = mean_powers
