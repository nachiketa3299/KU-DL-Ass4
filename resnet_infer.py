import tensorflow as tf
import data_helpers as dh
import numpy as np


# 이부분에 사용할 모델의 Timestamp를 적으면 추론 시작
whatModelTimestamp = input("- 추론에 사용할 모델의 타임스탬프를 입력하세요.")

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{whatModelTimestamp}/checkpoints", "Checkpoint directory from training run")
FLAGS = tf.flags.FLAGS
dataset = dh.read_my_images("./example/*")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        X = graph.get_operation_by_name("X").outputs[0]
        Y = graph.get_operation_by_name("Y").outputs[0]
        predictions = graph.get_operation_by_name("logit/predictions").outputs[0]

        inference_result = sess.run(predictions, feed_dict={X: dataset})
        print("airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
        print(inference_result)
