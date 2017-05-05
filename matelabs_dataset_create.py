import tensorflow as tf
import numpy as np
import os
from tensorflow.python.platform import gfile
def create_dataset(dir = 'Shapes/Database/Train' , image_file = 'datait.out' , label_file ='datalt.out'):
    """

         Args:
             dir : directory where the folders of images lie according to classes
             image_file: name of ouput images data file
             label_file: name of output labels  data file

         Returns:
             The return value. True for success, False otherwise.

     """


    sess=tf.Session()
    sub_dirs = [x[0] for x in gfile.Walk(dir)]
    del sub_dirs[0]
    data = []
    labels_list = np.empty(shape=(0))

    images_list = np.empty(shape=(1,784))
    for sub_dir in sub_dirs :
        print('Reading ..',sub_dir)
        file_list = []
        file_glob = os.path.join(sub_dir, '*.' + 'png')
        file_list.extend(gfile.Glob(file_glob))
        print(len(file_list))
        clas = np.array([sub_dirs.index(sub_dir)])
        for im in file_list:
            image = tf.read_file(im)
            image_tensor = tf.image.decode_png(image, channels=1)
            image_tensor=tf.reshape(image_tensor ,[-1])
            y = image_tensor.eval(session=sess)
            labels_list = np.append(labels_list, clas,axis=0)
            images_list =  np.append(images_list, y[np.newaxis,:],axis=0)




    np.savetxt(label_file, labels_list, delimiter=',')

    np.savetxt(image_file,images_list, delimiter=',')

    return

if __name__ == "__main__" :
    create_dataset()