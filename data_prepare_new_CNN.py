import pathlib
import numpy as np
from keras.src.preprocessing.image import ImageDataGenerator


class Data(object):
    def __init__(self, train_path, test_path):
        self.batch = 256  # 批处理个数
        self.img_size = (224, 224)  # 图片缩放尺寸

        # 获得每个文件夹名，即类名
        data_dir = pathlib.Path(train_path)
        self.class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
        self.image_shape = (self.img_size[0], self.img_size[1], 3)      # 传参,无实际意义

        # 创建ImageDataGenerator对象
        train_datagen = ImageDataGenerator(rescale=1. / 255,        # 像素值数据归一化
                                           rotation_range=45,  # 随机旋转图像的角度范围
                                           width_shift_range=0.2,  # 随机水平平移图像的比例
                                           height_shift_range=0.2,  # 随机垂直平移图像的比例
                                           shear_range=0.2,  # 随机剪切强度
                                           zoom_range=0.2,  # 随机缩放范围
                                           horizontal_flip=True,  # 随机水平翻转
                                           fill_mode='nearest',  # 用于填充新创建像素的方法
                                           validation_split=0.1   # 设置验证集占总体的比例
                                           )
        test_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=45,       # 随机旋转图像的角度范围
                                           width_shift_range=0.2,   # 随机水平平移图像的比例
                                           height_shift_range=0.2,  # 随机垂直平移图像的比例
                                           shear_range=0.2,         # 随机剪切强度
                                           zoom_range=0.2,          # 随机缩放范围
                                           horizontal_flip=True,    # 随机水平翻转
                                           fill_mode='nearest'      # 用于填充新创建像素的方法
                                           )

        # 训练数据生成器，图像向量的格式为(batch_size, height, width, 3)，标签为独热编码
        self.train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch,
            color_mode="rgb",
            class_mode="categorical",
            subset='training'  # 指定数据集为训练集
        )

        self.validation_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch,
            color_mode="rgb",
            class_mode="categorical",
            subset='validation'  # 指定数据集为验证集
        )

        self.test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=self.img_size,
            batch_size=self.batch,
            color_mode="rgb",
            class_mode="categorical")
