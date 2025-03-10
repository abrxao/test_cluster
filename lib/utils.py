import tensorflow as tf

# Constantes para conversão entre linear e dB
_10 = tf.constant(10, dtype=tf.float32)
BASE_10 = tf.math.log(_10)


def linearToDB(x):
    """
    Converte um valor linear para dB.

    Parâmetros:
    - x (TensorFlow tensor): Valor em escala linear.

    Retorna:
    - TensorFlow tensor: Valor convertido para dB.
    """
    return _10 / BASE_10 * tf.math.log(x)


def dBToLinear(x):
    """
    Converte um valor em dB para escala linear.

    Parâmetros:
    - x (TensorFlow tensor): Valor em dB.

    Retorna:
    - TensorFlow tensor: Valor convertido para escala linear.
    """
    return _10 ** (x / _10)


def genRandomDistances(mean_distances, var, shape):
    """
    Gera distâncias aleatórias para usuários na simulação com base em uma
    distribuição normal.

    Parâmetros:
    - mean_distances (float ou TensorFlow tensor): Média das distâncias dos usuários em metros.
    - var (float ou TensorFlow tensor): Variância das distâncias.
    - shape (tuple): Formato desejado do tensor de distâncias.

    Retorna:
    - TensorFlow tensor: Tensor de distâncias aleatórias geradas.
    """
    batch_size = shape[0]
    num_users = shape[2]

    # Forma inicial caso as distâncias dos usuários mudem a cada batch
    initial_shape = (batch_size, 1, num_users, 1, 1, 1, 1)

    """
    Caso as distâncias dos usuários sejam fixas em todos os batches, use:
    initial_shape = (1, 1, num_users, 1, 1, 1, 1)
    
    Referência: https://nvlabs.github.io/sionna/api/channel.wireless.html#sionna.channel.tr38901.TDL
    """

    # Desvio padrão das distâncias
    std_dev = tf.sqrt(tf.constant(var, dtype=tf.float32))

    # Retorna um tensor de distâncias aleatórias
    return tf.random.normal(shape=initial_shape, mean=mean_distances, stddev=std_dev)


def calculatePathLoss(fc, distances, sc_params, shape):
    """
    Calcula a perda de percurso (Path Loss) em dB e converte para escala linear.

    A função recebe parâmetros de frequência, distâncias entre usuários,
    parâmetros do cenário e o formato do tensor de taps (amplitudes retornadas
    pelo canal). Retorna um tensor com o mesmo formato, contendo os valores
    de path loss em escala linear para multiplicação ponto a ponto.

    Parâmetros:
    - fc (float ou TensorFlow tensor): Frequência de operação em Hz.
    - distances (TensorFlow tensor): Distâncias entre usuários em metros.
    - sc_params (dict): Parâmetros do cenário, incluindo `clutter_loss` e `shadow_fading`.
    - shape (tuple): Formato do tensor de taps.

    Retorna:
    - TensorFlow tensor: Path loss convertido para escala linear e pronto para uso.
    """
    path_loss = (
        sc_params["clutter_loss"]  # Perda por obstrução (dB)
        # TODO: Raul [] - Linearizar toda a equação para evitar conversão de dB para linear e vice-versa
        + tf.random.normal(
            shape=distances.shape, mean=0.0, stddev=sc_params["shadow_fading"]
        )  # Atenuação por sombra
        # Cálculo da perda de percurso em espaço livre (Free Space Path Loss)
        + 20 / BASE_10 * tf.math.log(fc / 1e9)  # Conversão da frequência
        + 32.45  # Constante da equação de FSPL
        + 20 / BASE_10 * tf.math.log(distances)  # Conversão da distância
    )

    # Converte a perda de percurso de dB para linear
    pl_linear = tf.sqrt(tf.pow(10.0, -path_loss / 10.0))

    # Ajusta o tensor para ter o formato correto e retorna em número complexo
    return tf.broadcast_to(
        tf.complex(pl_linear, tf.zeros_like(pl_linear, dtype=tf.float32)), shape=shape
    )
