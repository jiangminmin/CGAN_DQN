import numpy as np
import tensorflow as tf
import collections


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def my_placeholder():
    Model = collections.namedtuple("Model", "my_inputs, y")

    my_inputs = tf.placeholder(dtype=tf.uint8,shape=(100,3202),name="inputs")

    my_img = my_inputs[:, :, tf.newaxis]
    my_img = tf.image.decode_jpeg(tf.image.encode_jpeg(my_img))
    my_img = tf.image.convert_image_dtype(my_img, dtype=tf.float32)
    my_img = tf.identity(my_img)
    my_img.set_shape([None, None, 1])


    inputs = preprocess(my_img[:, 1601:, :])
    targets = preprocess(my_img[:, :1601, :])
    inputs = tf.image.resize_images(inputs, [256, 256], method=tf.image.ResizeMethod.AREA)
    targets = tf.image.resize_images(targets, [256, 256], method=tf.image.ResizeMethod.AREA)
    inputs = inputs[tf.newaxis,:,:,:]
    print(inputs.shape)
    #inputs, targets = tf.train.batch([inputs, targets], batch_size=1)


    return Model(
        my_inputs = my_inputs,
        y = inputs,
    )

model = my_placeholder()
x = np.ones(shape=(100,3202),dtype=np.uint8)
with tf.Session() as sess:
    print(sess.run(model.y,feed_dict={model.my_inputs:x}))