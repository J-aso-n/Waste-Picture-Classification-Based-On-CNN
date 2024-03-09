
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.utils import plot_model


class ClassificationModel(object):
    def __init__(self, class_names, image_shape):   # 输入参数：label数组，图片参数
        self.filters = 32                # 卷积核数量
        self.kernel_size = 3            # 卷积核大小
        self.activation = 'relu'        # 激活函数

        self.pool_size = 2              # 池化层大小
        self.epochs = 25                # 训练的批次数

        # CNN卷积模型
        self.model = tf.keras.models.Sequential([                   # 顺序模型
            # tf.keras.layers.Conv2D(filters=5,  # 卷积层。这里使用 5 个卷积核
            #                        kernel_size=self.kernel_size,  # 卷积核的大小为 3x3
            #                        activation=self.activation,  # 激活函数
            #                        input_shape=image_shape),
            # tf.keras.layers.MaxPool2D(pool_size=2,  # 最大池化层，大小为 2x2
            #                           padding='valid'),  # 不补零
            # tf.keras.layers.Flatten(),  # 将卷积层输出的特征图展平为一维向量
            # tf.keras.layers.Dense(len(class_names), activation="softmax")  # 全连接层。输出层的激活函数为 softmax，用于多类别分类任务
            tf.keras.layers.Conv2D(filters=self.filters,            # 卷积层。这里使用 5 个卷积核
                                   kernel_size=self.kernel_size,    # 卷积核的大小为 3x3
                                   activation=self.activation,      # 激活函数
                                   input_shape=image_shape),
            tf.keras.layers.MaxPool2D(pool_size=2,                  # 最大池化层，大小为 2x2
                                      padding='valid'),             # 不补零
            tf.keras.layers.Conv2D(filters=16,  # 卷积层。这里使用 5 个卷积核
                                   kernel_size=self.kernel_size,  # 卷积核的大小为 3x3
                                   activation=self.activation,  # 激活函数
                                   input_shape=(49,49,5)),
            tf.keras.layers.MaxPool2D(pool_size=2,  # 最大池化层，大小为 2x2
                                      padding='valid'),  # 不补零
            tf.keras.layers.Conv2D(filters=32,  # 卷积层。这里使用 5 个卷积核
                                   kernel_size=self.kernel_size,  # 卷积核的大小为 3x3
                                   activation=self.activation,  # 激活函数
                                   input_shape=(23, 23, 3)),
            tf.keras.layers.MaxPool2D(pool_size=2,  # 最大池化层，大小为 2x2
                                      padding='valid'),  # 不补零
            tf.keras.layers.Flatten(),                              # 将卷积层输出的特征图展平为一维向量
            tf.keras.layers.Dense(256, activation="relu"),  # 全连接层。输出层的激活函数为 softmax，用于多类别分类任务
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),  # 全连接层。输出层的激活函数为 softmax，用于多类别分类任务
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(class_names), activation="sigmoid"),  # 全连接层。输出层的激活函数为 softmax，用于多类别分类任务

        ])
        # plot_model(self.model)
        # plt.show()

        # 编译模型，配置损失函数、优化器和评估指标
        self.model.compile(loss="binary_crossentropy",  # 类别分类问题常用的损失函数，适用于输出是独热编码形式的标签categorical_crossentropy
                        optimizer=tf.keras.optimizers.Adam(),  # Adam优化器
                        metrics=['accuracy'])  # 在训练过程中监控模型的准确率

        print(self.model.summary())

    def train(self, train_ds, val_ds):
        # 定义 EarlyStopping 回调,当准确率不会改变了停止训练
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy',  # 监控准确率
            patience=3,  # 如果连续 3 个 epoch 准确率没有改善，则停止训练
            restore_best_weights=True  # 恢复最佳权重
        )

        # 训练模型
        train_result = self.model.fit(train_ds,
                                epochs=self.epochs,
                                validation_data=val_ds,
                                callbacks=[early_stopping]
                                )
        pd.DataFrame(train_result.history).plot()  # 展示训练结果
        plt.show()

    def run(self):
        return