import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data

import os
import presets_and_results as pr
import pickle

pickles = os.listdir(os.path.join(os.path.curdir, 'INFOS'))
for pckl in pickles:
    with open(pckl, 'rb') as pR:
        P = pickle.load(pR)
        if not P.finished:
            (x_train_val, y_train_val), (x_test, y_test) = load_data()

            # Eval Parameters
            pr.Preset().del_all_flags(tf.flags.FLAGS)
            tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{str(P.timestamp)}/checkpoints", "Checkpoint directory from training run")

            FLAGS = tf.flags.FLAGS
            # ==================================================
            y_test_one_hot = np.eye(10)[y_test]
            y_test_one_hot = np.squeeze(y_test_one_hot, axis=1)
            x_test = (x_test/127.5) - 1

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
                    accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

                    test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test_one_hot})
                    P.test_accuracy = test_accuracy
                    print('- Preset           :', P.preset)
                    print('- Timestamp        :', P.test_accuracy)
                    print('- Test Max Accuracy:', test_accuracy)
                    P.finished = True
        with open('eval_result.txt', 'w') as f:
            s = ''
            for key in P.__dict__.keys():
                s += P.__dict__[key] + '\t'
            f.write(s)

        with open(pckl, 'wb') as pW:
            pickle.dump(P, pW)
