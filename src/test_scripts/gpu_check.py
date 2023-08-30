# # import tensorflow as tf
# # from tensorflow.python.client import device_lib
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
#
# # print(device_lib.list_local_devices())
# print(tf.config.list_physical_devices())
#
#
#
# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
#
# # Creates a session with log_device_placement set to True.
# config=tf.ConfigProto(log_device_placement=True)
# sess = tf.Session(config=config)
# #config gpu list
#
# # Runs the op.
# print(sess.run(c))


import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(device)