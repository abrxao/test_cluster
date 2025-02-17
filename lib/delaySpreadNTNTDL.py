import numpy as np

delay_spread_data = {
    "los": {
        "dense_urban": {
            "s_band": {
                "mu_lgDS": [
                    -7.12,
                    -7.28,
                    -7.45,
                    -7.73,
                    -7.91,
                    -8.14,
                    -8.23,
                    -8.28,
                    -8.36,
                ]
            },
            "ka_band": {
                "mu_lgDS": [
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
            },
        },
    },
    "nlos": {
        "dense_urban": {
            "s_band": {
                "mu_lgDS": [
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
            },
            "sa_band": {
                "mu_lgDS": [
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
            },
        },
    },
}


def getDelaySpread(is_los: bool, zone: str, band: str, angle_of_arrival: float):
    if angle_of_arrival < 10 or angle_of_arrival > 90:
        raise ValueError("Angle of arrival must be between 10 and 90 degrees")
    if zone not in delay_spread_data[is_los]:
        raise ValueError(f"Zone {zone} not found")
    if band not in delay_spread_data[is_los][zone]:
        raise ValueError(f"Band {band} not found")

    # Getting los key
    los_key = "los" if is_los else "nlos"
    # Acess the file with the delay spread data
    delay_spread_list = delay_spread_data[los_key][zone][band]
    # Get the index of the angle of arrival
    idx = np.round(angle_of_arrival / 10 - 1).astype(int)
    # Convert the angle of arrival to radians
    return 10 ** (delay_spread_list["mu_lgDS"][idx])
