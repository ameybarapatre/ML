import tensorflow as tf
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn import datasets, svm, metrics

def matelabs_shapes(train = True , checkpoint = 'matelabs_model.ckpt', train_image = 'datait.out' ,train_label ='datalt.out' ,
                    test_image = 'datai.out' , test_label = 'datal.out'):
    """Training set (train_image , train_label ) is required  to fit normalization over testing set
        default data set is data.out and d.out created using  create_dataset() from matelabs_dataset_create from
        the 'Shapes/Database/Train'  folder

        Args:
            train: True to train the CNN , False to test it.
            checkpoint: checkpoint file name.
            train_image:training set  of numpy array of images
            train_label:training set  of corresponding training labels
            test_image:test set of numpy array of images
            test_label:test set of numpy array of labels corresponding to test images

        Returns:
            The return value. True for success, False otherwise.

    """
    tf.reset_default_graph()
    with tf.Session() as session :
        x = tf.placeholder(tf.float32, shape=[None,784 ])
        y_ = tf.placeholder(tf.float32, shape=[None])


        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)

        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)

        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)


        W_conv3 = weight_variable([5, 5, 64, 64])
        b_conv3 = bias_variable([64])

        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
        norm2 = tf.nn.lrn(h_conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

        h_pool2 = max_pool_2x2(norm2)

        W_conv4 = weight_variable([5, 5, 64, 64])
        b_conv4 = bias_variable([64])

        h_conv4 = tf.nn.relu(conv2d(h_pool2, W_conv4) + b_conv4)

        W_fc1 = weight_variable([7 * 7 * 64, 50])
        b_fc1 = bias_variable([50])

        h_pool2_flat = tf.reshape(h_conv4, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([50,5])
        b_fc2 = bias_variable([5])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2




        cross_entropy = tf.divide(tf.reduce_sum(tf.contrib.losses.sparse_softmax_cross_entropy(labels=tf.cast(y_ , tf.int32),logits= y_conv)) , 50 )
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
        taccuracy= tf.equal(tf.argmax(y_conv,1), tf.cast(y_ , tf.int64))

        accuracy =  tf.divide( tf.reduce_sum(tf.cast(taccuracy , tf.int32))  , 100 )
        predicted = tf.argmax(y_conv,1)

        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())


        image_data = np.loadtxt(train_image, delimiter=',' , dtype='float32')
        label_data = np.loadtxt(train_label , delimiter =',' , dtype='float32')

        print( 'Preprocessing.... ' ,len(image_data), len(label_data))


        # centering and normalizing the data

        scaler = preprocessing.StandardScaler()
        scaler.fit(image_data)
        image_data = scaler.transform(image_data)
        x1_n = image_data
        def shuffle_in_unison(a, b):
            assert len(a) == len(b)
            shuffled_a = np.empty(a.shape, dtype=a.dtype)
            shuffled_b = np.empty(b.shape, dtype=b.dtype)
            permutation = np.random.permutation(len(a))
            for old_index, new_index in enumerate(permutation):
                shuffled_a[new_index] = a[old_index]
                shuffled_b[new_index] = b[old_index]
            return shuffled_a, shuffled_b


        x1, y2 = x1_n, label_data
        if os.path.exists("checkpoint")!=True:

            print("Training..." , len(x1_n))
            for j in range(0,4):
                for  i in range(0,int(len(x1_n)/50)):

                    train_step.run(feed_dict={x: x1[i*50:(i+1)*50] , y_: y2[i*50:(i+1)*50], keep_prob: 0.1})
                    train_accuracy = cross_entropy.eval(feed_dict={x: x1[i*50:(i+1)*50] , y_: y2[i*50:(i+1)*50], keep_prob: 1.0})
                    print(train_accuracy)

            saver.save(session, "./")
        else :

            image_data = np.loadtxt(test_image, delimiter=',' , dtype='float32')
            label_data = np.loadtxt(test_label , delimiter =',' , dtype='float32')
            image_data = scaler.transform(image_data)


            x1, y2 = image_data, label_data


            saver.restore(session, "./")
            print("Testing..." ,len(x1))

            acc=predicted.eval(feed_dict={x: x1, y_: y2, keep_prob: 1.0} )
            print(metrics.confusion_matrix(y2, acc))
            print(metrics.classification_report(y2, acc))

if __name__ == '__main__':
    matelabs_shapes()
    print('Saving')
    #matelabs_shapes(train=False)
