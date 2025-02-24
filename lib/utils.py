import tensorflow as tf

_10 = tf.constant(10, dtype=tf.float32)
BASE_10 = tf.math.log(_10)


def linearToDB(x):
    return _10 / BASE_10 * tf.math.log(x)


def dBToLinear(x):
    return _10 ** (x / _10)


def genRandomDistances(mean_distances, var, shape):
    batch_size = shape[0]
    num_users = shape[2]
    # Initial shape if user distances in simulation change for each batch
    initial_shape = (batch_size, 1, num_users, 1, 1, 1, 1)
    """ Initial shape if user distances in simulation are the same for each batch
    initial_shape = (1, 1, num_users, 1, 1, 1, 1) """
    # Distances Standard deviation
    std_dev = tf.sqrt(tf.constant(var, dtype=tf.float32))
    # Return random distances
    return tf.random.normal(shape=initial_shape, mean=mean_distances, stddev=std_dev)


def calculatePathLoss(fc, distances, sc_params, shape):
    """
    Essa função calcula o path loss em dB recebendo paramentros de frequência,
    distâncias, parametros de cenário e shape do tensor "a" que é o tensor de
    amplitude dos taps retornado quando o canal é criado. Após isso, será retornado
    um tensor com o mesmo shape de "a" com os valores de path loss em linear para
    ser efetuada a multiplicação de Hadamard (produto ponto a ponto) com "a".

    fc: Frequência de operação em Hz
    distances: Distâncias entre usuários em metros
    sc_params: Parametros de cenário
    shape: Shape do tensor de taps
    """
    path_loss = (
        # Calculate path loss in dB considering shadowing and clutter
        sc_params["clutter_loss"]
        + sc_params["shadow_fading"]
        # Calculate Free Space Path Loss in dB
        + 20 / BASE_10 * tf.math.log(fc / 1e9)
        + 32.45
        + 20 / BASE_10 * tf.math.log(distances)
    )
    # Convert path loss to linear
    pl_linear = tf.sqrt(
        tf.pow(
            10.0,
            -(path_loss) / 10.0,
        )
    )
    return tf.broadcast_to(
        tf.complex(pl_linear, tf.zeros_like(pl_linear, dtype=tf.float32)), shape=shape
    )
