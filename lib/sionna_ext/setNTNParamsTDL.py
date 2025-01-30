import json
from importlib_resources import files
import numpy as np

import tensorflow as tf

from sionna import PI, SPEED_OF_LIGHT
from sionna import config
from sionna.utils import (
    insert_dims,
    expand_to_rank,
    matrix_sqrt,
    split_dim,
    flatten_last_dims,
)
from sionna.channel import ChannelModel

from . import models
from sionna.channel.tr38901 import TDL


def setNTNParamsTDL(channel: TDL, fname: str):
    source = files(models).joinpath(f"TDL-{fname}.json")
    # pylint: disable=unspecified-encoding
    with open(source) as parameter_file:
        params = json.load(parameter_file)

    # LoS scenario ?
    channel._los = bool(params["los"])

    # Scale the delays
    channel._scale_delays = bool(params["scale_delays"])

    # Loading cluster delays and mean powers
    channel._num_clusters = tf.constant(params["num_clusters"], tf.int32)

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
