import tensorflow as tf
import os


def configGPUTensorflow():
    print("\nConfigurando GPU...\n")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Avoid warnings from TensorFlow
    tf.get_logger().setLevel("ERROR")
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print("__________________")
        print("\n GPU Reconhecida ")
        print("__________________")
        gpu_num = 0  # Index of the GPU to be used
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
        try:
            # tf.config.set_visible_devices([], 'GPU')
            tf.config.set_visible_devices(gpus[gpu_num], "GPU")
            tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
        except RuntimeError as e:
            print(e)
    input("Aperte Enter para prosseguir")
    return tf
