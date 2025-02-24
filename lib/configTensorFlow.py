import tensorflow as tf
import os


def configTensorflow(useGPU: bool):
    if useGPU == False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], "GPU")
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != "GPU"
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
        return
    
    # Avoid warnings from TensorFlow
    tf.get_logger().setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Index of the GPU to be used
    gpu_num = 0  # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_num], "GPU")
            tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
        except RuntimeError as e:
            print(e)
    return tf
