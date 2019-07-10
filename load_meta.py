import tensorflow as tf

saver = tf.train.import_meta_graph("facades_train/model-200000.meta")
checkpoint = tf.train.latest_checkpoint("facades_train")
graph = tf.get_default_graph
with tf.Session() as sess:
    saver.restore(sess,checkpoint)
    output=sess.run(graph.get_tensor_by_name("outputs"))
    print(output)
"""
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')
"""