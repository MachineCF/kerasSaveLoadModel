# 导入模块
import os
import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Activation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 读取CSV数据集，并拆分为训练集和测试集
# 该函数的传入参数为CSV_FILE_PATH: csv文件路径
def load_data(CSV_FILE_PATH):
    IRIS = pd.read_csv(CSV_FILE_PATH)
    target_var = 'class'  # 目标变量
    # 数据集的特征
    features = list(IRIS.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = IRIS[target_var].unique()
    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target, 将目标变量进行编码
    IRIS['target'] = IRIS[target_var].apply(lambda x: Class_dict[x])
    # 对目标变量进行0-1编码(One-hot Encoding)
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(IRIS['target'])
    y_bin_labels = []  # 对多分类进行0-1编码的变量
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        IRIS['y' + str(i)] = transformed_labels[:, i]
    # 将数据集分为训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(IRIS[features], IRIS[y_bin_labels], \
                                                        train_size=0.7, test_size=0.3, random_state=0)
    return train_x, test_x, train_y, test_y, Class_dict


def main():
    # 0. 开始
    print("\nIris dataset using Keras/TensorFlow ")
    np.random.seed(4)
    tf.set_random_seed(10)

    # 1. 读取CSV数据集
    print("Loading Iris data into memory")
    CSV_FILE_PATH = './iris.csv'
    train_x, test_x, train_y, test_y, Class_dict = load_data(CSV_FILE_PATH)

    # 2.定义模型
    init = K.initializers.glorot_uniform(seed=1)
    model = Sequential()
    model.add(Dense(20, input_shape=(4,), kernel_initializer=init, activation='relu', name='dense_1'))
    model.add(Dense(10, kernel_initializer=init, activation='relu', name='dense_2'))
    model.add(Dense(10, kernel_initializer=init, activation='relu', name='dense_3'))
    model.add(Dense(10, kernel_initializer=init, activation='relu', name='dense_4'))
    model.add(Dense(3, kernel_initializer=init, activation='softmax', name='output'))
    model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam(), metrics=['accuracy'])

    # 3. 训练模型同时保存效果最好的模型
    batchSize = 5
    Epochs = 20
    print("Starting training ")
    filepath = "weightsBest.hdf5"
    # 保存效果最好的模型
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(train_x, train_y, validation_data=(test_x, test_y),
              batch_size=batchSize, epochs=Epochs, shuffle=True,
              verbose=2, callbacks=callbacks_list)
    print("Training finished \n")


if __name__ == '__main__':
    main()
