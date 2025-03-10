import numpy as np
import tensorflow as tf
from consts import tdl_los, scenarios_params


def getBand(fc: float):
    """
    Determina a banda (S ou Ka) com base na frequência de operação.

    Parâmetros:
    - fc (float): Frequência do canal em Hz.

    Retorna:
    - str: "s_band" se a frequência estiver na banda S,
           "ka_band" se estiver na banda Ka,
           None se não pertencer a nenhuma das bandas definidas.
    """
    # Verifica se a frequência está dentro da faixa da banda S (2 a 4 GHz)
    if 2e9 <= fc <= 4e9:
        return "s_band"
    # Verifica se a frequência está dentro da faixa da banda Ka (27 a 40 GHz)
    elif 27e9 <= fc <= 40e9:
        return "ka_band"
    # Retorna None se a frequência não pertencer a nenhuma das bandas
    return None


def getScenarioParams(model: str, zone: str, fc: float, elevation_angle: float) -> dict:
    """
    Retorna os parâmetros do cenário de propagação com base no modelo, zona,
    frequência e ângulo de elevação.

    Parâmetros:
    - model (str): Modelo de propagação utilizado.
    - zone (str): Zona do cenário (ex: "urban", "suburban", "rural").
    - fc (float): Frequência do canal em Hz.
    - elevation_angle (float): Ângulo de elevação em graus.

    Retorna:
    - dict: Dicionário contendo os parâmetros do cenário:
        - delay_spread (TensorFlow tensor): Espalhamento de atraso (em segundos).
        - shadow_fading (TensorFlow tensor): Atenuação por sombra (em dB).
        - clutter_loss (TensorFlow tensor): Perda por obstrução (em dB).

    Lança:
    - ValueError: Caso o ângulo de elevação esteja fora do intervalo permitido
                  (10° a 90°), a frequência não esteja em uma banda válida,
                  ou a zona informada não esteja nos parâmetros dos cenários.
    """
    # Verifica se o ângulo de elevação está dentro do intervalo permitido (10° a 90°)
    if not (10 <= elevation_angle <= 90):
        raise ValueError("O ângulo de elevação deve estar entre 10 e 90 graus")

    # Determina se o modelo é LOS (linha de visada) ou NLOS (não linha de visada)
    is_los = tdl_los[model]
    los_key = "los" if is_los else "nlos"

    # Determina a banda com base na frequência
    band = getBand(fc)
    if band is None:
        raise ValueError("Frequência fora das bandas S e Ka")

    # Verifica se a zona especificada está nos parâmetros dos cenários
    if zone not in scenarios_params[los_key]:
        raise ValueError(f"Zona '{zone}' não encontrada nos parâmetros do cenário")

    # Obtém os parâmetros do cenário para a combinação LOS/NLOS, zona e banda
    params_list = scenarios_params[los_key][zone][band]

    # Calcula o índice correspondente ao ângulo de chegada (cada 10° um índice)
    idx = np.round(elevation_angle / 10 - 1).astype(int)

    # Obtém o espalhamento de atraso (log10 de segundos) e converte para segundos
    delay_spread = 10 ** params_list["delay_spread"][idx]

    # Obtém a atenuação por sombra (shadow fading) em dB
    shadow_fading = params_list["shadow_fading"][idx]

    # Obtém a perda por obstrução (clutter loss) em dB (0 para LOS)
    clutter_loss = 0 if is_los else params_list["clutter_loss"][idx]

    return {
        "delay_spread": tf.constant(delay_spread, dtype=tf.float32),
        "shadow_fading": tf.constant(shadow_fading, dtype=tf.float32),
        "clutter_loss": tf.constant(clutter_loss, dtype=tf.float32),
    }
