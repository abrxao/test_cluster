import json
from importlib_resources import files
import numpy as np
import tensorflow as tf
from . import models
from sionna.channel.tr38901 import TDL
from ..delaySpreadNtnTdlData import delay_spread_data


def setNTNParamsTDL(
    channel: TDL, fname: str, zone: str, band: str, angle_of_arrival: float = 50
):
    """Set the parameters of the TDL channel model for NTN scenarios.
    Zone: Dense Urban
    band: s_band, ka_band
    fname: A, B, C, D
    angle: Angle of arrival in degrees
    """

    source_base = files(models).joinpath(f"TDL-{fname}.json")
    # pylint: disable=unspecified-encoding
    with open(source_base) as parameter_file:
        params = json.load(parameter_file)

    # LoS scenario ?
    channel._los = bool(params["los"])

    # Scale the delays
    channel._scale_delays = bool(params["scale_delays"])

    # Loading cluster delays and mean powers
    channel._num_clusters = tf.constant(params["num_clusters"], tf.int32)

    # Getting los key
    los_key = "los" if channel._los else "nlos"
    # Acess the file with the delay spread data
    delay_spread_list = delay_spread_data[los_key][zone][band]
    # Get the index of the angle of arrival
    idx = 0
    # Convert the angle of arrival to radians
    angle_of_arrival = np.deg2rad(angle_of_arrival)
    for i, theta in enumerate(delay_spread_list["theta_C"]):
        if angle_of_arrival >= theta:
            idx = i
            break
    # Setting delay spread
    channel._delay_spread = 10 ** (delay_spread_list["mu_lgDS"][idx])
    # Retrieve power and delays
    delays = tf.constant(params["delays"], channel._real_dtype)
    mean_powers = np.power(10.0, np.array(params["powers"]) / 10.0)
    mean_powers = tf.constant(mean_powers, channel._dtype)
    if channel._los:
        # The power of the specular component of the first path is stored
        # separately
        channel._los_power = mean_powers[0]
        mean_powers = mean_powers[1:]
        # The first two paths have 0 delays as they correspond to the
        # specular and reflected components of the first path.
        # We need to keep only one.
        delays = delays[1:]

    # Normalize the PDP
    if channel._los:
        norm_factor = tf.reduce_sum(mean_powers) + channel._los_power
        channel._los_power = channel._los_power / norm_factor
        mean_powers = mean_powers / norm_factor
    else:
        norm_factor = tf.reduce_sum(mean_powers)
        mean_powers = mean_powers / norm_factor

    channel._delays = delays
    channel._mean_powers = mean_powers
