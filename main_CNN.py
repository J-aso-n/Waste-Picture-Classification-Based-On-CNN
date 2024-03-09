# dataset: https://www.kaggle.com/datasets/techsash/waste-classification-data/data
# waste/TRAIN/
# waste/TEST/

# import data_prepare_CNN
import data_prepare_new_CNN
import model_CNN


def main():
    train_path = "waste/TRAIN/"
    test_path = "waste/TEST/"

    # Data = data_prepare_CNN.Data(train_path, test_path)     # 数据处理模型
    Data = data_prepare_new_CNN.Data(train_path, test_path)     # 新的数据处理模型
    model = model_CNN.ClassificationModel(Data.class_names,Data.image_shape)  # CNN分类模型

    model.train(Data.train_generator, Data.test_generator)
    # model.train(Data.train_ds, Data.val_ds)     # 模型训练


if __name__ == '__main__':    main()