import numpy as np
import tensorflow as tf

""" Esse mapa foi criado de acordo com a TR 38.811 e todos os valores estão em dB.
    Exceto o delay spread que está em log10 de segundos.
"""
scenarios_params = {
    "los": {
        "dense_urban": {
            "s_band": {
                "delay_spread": [
                    -7.12,
                    -7.28,
                    -7.45,
                    -7.73,
                    -7.91,
                    -8.14,
                    -8.23,
                    -8.28,
                    -8.36,
                ],
                "shadow_fading": [3.5, 3.4, 2.9, 3.0, 3.1, 2.7, 2.5, 2.3, 1.2],
            },
            "ka_band": {
                "delay_spread": [
                    -7.43,
                    -7.62,
                    -7.76,
                    -8.02,
                    -8.13,
                    -8.30,
                    -8.34,
                    -8.39,
                    -8.45,
                ],
                "shadow_fading": [2.9, 2.4, 2.7, 2.4, 2.4, 2.7, 2.6, 2.8, 0.6],
            },
        }
    },
    "nlos": {
        "dense_urban": {
            "s_band": {
                "delay_spread": [
                    -6.84,
                    -6.81,
                    -6.94,
                    -7.14,
                    -7.34,
                    -7.53,
                    -7.67,
                    -7.82,
                    -7.84,
                ],
                "shadow_fading": [15.5, 13.9, 12.4, 11.7, 10.6, 10.5, 10.1, 9.2, 9.2],
                "clutter_loss": [34.3, 30.9, 29.0, 27.7, 26.8, 26.2, 25.8, 25.5, 25.5],
            },
            "ka_band": {
                "delay_spread": [
                    -6.86,
                    -6.84,
                    -7.00,
                    -7.21,
                    -7.42,
                    -7.86,
                    -7.76,
                    -8.07,
                    -7.95,
                ],
                "shadow_fading": [17.1, 17.1, 15.6, 14.6, 14.2, 12.6, 12.1, 12.3, 12.3],
                "clutter_loss": [44.3, 39.9, 37.5, 35.8, 34.6, 33.8, 33.3, 33.0, 32.9],
            },
        }
    },
}

# TDL models los key
tdl_los = {
    "A": False,
    "B": False,
    "C": True,
    "D": True,
}


# Setting band according to fc
def getBand(fc: float):
    # Check if fc is S band
    if 2e9 >= fc or fc <= 4e9:
        return "s_band"
    # Check if fc is Ka band
    elif 27e9 >= fc or fc <= 40e9:
        return "ka_band"
    # If fc is not in any band
    return None


def getScenarioParams(model: str, zone: str, fc: float, angle_of_arrival: float):
    if angle_of_arrival < 10 or angle_of_arrival > 90:
        raise ValueError("Angle of arrival must be between 10 and 90 degrees")
    # Getting los key
    is_los = tdl_los[model]
    los_key = "los" if is_los else "nlos"
    # Getting band
    band = getBand(fc)
    if band is None:
        raise ValueError("Band not found")
    # Check if zone is in the scenarios params
    if zone not in scenarios_params[los_key]:
        raise ValueError(f"Zone {zone} not found")

    # Acess the file with params for the scenario
    params_list = scenarios_params[los_key][zone][band]
    # Get the index of the angle of arrival
    idx = np.round(angle_of_arrival / 10 - 1).astype(int)
    # Get delay spread in log10 of seconds and convert to seconds
    delay_spread = 10 ** (params_list["delay_spread"][idx])
    # Get shadow fading in dB
    shadow_fading = params_list["shadow_fading"][idx]
    # Get clutter loss in dB
    clutter_loss = 0 if is_los else params_list["clutter_loss"][idx]
    return {
        "delay_spread": tf.constant(delay_spread, dtype=tf.float32),
        "shadow_fading": tf.constant(shadow_fading, dtype=tf.float32),
        "clutter_loss": tf.constant(clutter_loss, dtype=tf.float32),
    }
