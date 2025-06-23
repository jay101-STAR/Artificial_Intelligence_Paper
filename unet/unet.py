import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 配置环境变量以减少TensorFlow的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入必要的库用于深度学习、图像处理和可视化

# 添加GPU检测与配置
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU 可用: {tf.config.list_physical_devices('GPU')}")
# 检测GPU是否可用
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存增长，按需分配内存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"使用 GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("警告：未检测到GPU，将使用CPU运行")

# 定义Dice损失函数，用于评估分割模型的性能
def dice_loss(y_true, y_pred):
    smooth = 1.0  # 平滑因子，防止除以零的情况
    intersection = 2.0 * K.sum(K.abs(y_true * y_pred)) + smooth
    union = K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth
    return 1 - (intersection / union)  # 返回Dice系数的补数作为损失

# 定义加权二元交叉熵损失函数
def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * weights[1] + (1. - y_true) * weights[0]
        weighted_bce = weight_vector * bce
        return K.mean(weighted_bce)
    return loss

# 定义焦点损失函数，用于处理类别不平衡问题
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        alpha_t = tf.where(tf.equal(y_true, 1), alpha * tf.ones_like(y_pred), (1-alpha) * tf.ones_like(y_pred))
        fl = -alpha_t * K.pow((1-pt_1), gamma) * K.log(pt_1) - (1-alpha_t) * K.pow(pt_0, gamma) * K.log(1.-pt_0)
        return K.mean(fl)
    return loss

# 定义组合损失函数，结合Dice损失、加权二元交叉熵和焦点损失
def combined_loss(dice_weight=0.5, wbce_weight=0.3, focal_weight=0.2):
    def loss(y_true, y_pred):
        dice = dice_loss(y_true, y_pred)
        wbce = weighted_binary_crossentropy([0.1, 0.9])(y_true, y_pred)
        focal = focal_loss()(y_true, y_pred)
        return dice_weight * dice + wbce_weight * wbce + focal_weight * focal
    return loss

# 定义U-Net模型类
class U_Net():
    def __init__(self):
        self.height = 256  # 图像高度
        self.width = 256   # 图像宽度
        self.channels = 1  # 图像通道数（灰度图）
        self.shape = (self.height, self.width, self.channels)  # 输入图像形状
        optimizer = Adam(0.002, 0.5)  # Adam优化器，学习率为0.002
        self.unet = self.build_unet()  # 构建U-Net模型
        self.unet.compile(loss=dice_loss,  # 使用Dice损失函数
                          optimizer=optimizer,
                          metrics=[self.metric_fun])  # 自定义评估指标
        self.unet.summary()  # 打印模型结构摘要

    # 构建U-Net模型结构
    def build_unet(self, n_filters=16, dropout=0.1, batchnorm=True, padding='same'):
        def conv2d_block(input_tensor, n_filters=16, kernel_size=3, batchnorm=True, padding='same'):
            x = Conv2D(n_filters, kernel_size, padding=padding)(input_tensor)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(n_filters, kernel_size, padding=padding)(x)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x

        img = Input(shape=self.shape)  # 输入层
        c1 = conv2d_block(img, n_filters=n_filters * 1, padding=padding, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)  # 最大池化层
        p1 = Dropout(dropout * 0.5)(p1)  # Dropout层

        c2 = conv2d_block(p1, n_filters=n_filters * 2, padding=padding, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters * 4, padding=padding, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters * 8, padding=padding, batchnorm=batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(p4, n_filters=n_filters * 16, padding=padding, batchnorm=batchnorm)

        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)  # 反卷积层
        u6 = concatenate([u6, c4])  # 特征图拼接
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters * 8, padding=padding, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters * 4, padding=padding, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, padding=padding, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters * 1, padding=padding, batchnorm=batchnorm)

        output = Conv2D(1, (1, 1), activation='sigmoid')(c9)  # 输出层，使用Sigmoid激活函数
        return Model(img, output)  # 返回构建好的模型

    # 自定义评估指标函数
    def metric_fun(self, y_true, y_pred):
        fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, 0.1), tf.float32)) + 1e-8
        return fz / fm

    # 加载训练数据和标签
    def load_data(self):
        x_train = []
        x_label = []
        train_files = []
        label_files = []
        # 加载训练图像
        for file in glob('./train/*'):
            for filename in glob(file + '/*'):
                img = np.array(Image.open(filename), dtype='float32') / 255  # 将图像归一化到[0,1]
                x_train.append(img[256:, 128:384])  # 裁剪图像
                train_files.append(filename)
        # 加载标签图像
        for file in glob('./label/*'):
            for filename in glob(file + '/*'):
                img = np.array(Image.open(filename), dtype='float32') / 255
                x_label.append(img[256:, 128:384])
                label_files.append(filename)
        # 将列表转换为NumPy数组并添加通道维度
        x_train = np.expand_dims(np.array(x_train), axis=3)
        x_label = np.expand_dims(np.array(x_label), axis=3)
        # 打乱数据顺序
        np.random.seed(116)
        np.random.shuffle(x_train)
        np.random.seed(116)
        np.random.shuffle(x_label)
        # 将数据集划分为训练集和验证集
        return x_train[:2700, :, :], x_label[:2700, :, :], x_train[2700:, :, :], x_label[2700:, :, :], train_files, label_files

    # 训练模型
    def train(self, epochs=101, batch_size=32):
        os.makedirs('./weights', exist_ok=True)  # 创建权重保存目录
        os.makedirs('./evaluation/preprocessing', exist_ok=True)  # 创建预处理结果保存目录

        x_train, x_label, y_train, y_label, train_files, label_files = self.load_data()

        # 可视化预处理前后对比
        sample_idx = 2
        original_path = train_files[sample_idx]
        label_path = label_files[sample_idx]

        original_img = np.array(Image.open(original_path)) / 255.0
        original_label = np.array(Image.open(label_path)) / 255.0
        processed_img = x_train[sample_idx, :, :, 0]
        processed_label = x_label[sample_idx, :, :, 0]

        # 绘制图像对比
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 3, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(processed_img, cmap='gray')
        plt.title("Processed Image\n(Cropped & Normalized)")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(original_label, cmap='gray')
        plt.title("Original Label")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(processed_label, cmap='gray')
        plt.title("Processed Label")
        plt.axis('off')

        # 绘制直方图对比
        plt.subplot(2, 3, 3)
        plt.hist(original_img.flatten(), bins=50, alpha=0.5, color='blue', label='Original')
        plt.hist(processed_img.flatten(), bins=50, alpha=0.5, color='orange', label='Processed')
        plt.legend()
        plt.title("Pixel Histogram")

        plt.subplot(2, 3, 6)
        plt.hist(original_label.flatten(), bins=50, alpha=0.5, color='blue', label='Original')
        plt.hist(processed_label.flatten(), bins=50, alpha=0.5, color='orange', label='Processed')
        plt.legend()
        plt.title("Label Histogram")

        plt.tight_layout()
        plt.savefig('./evaluation/preprocessing/preprocessing_comparison.png')
        plt.close()

        # 绘制箱线图对比
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.boxplot(data=[original_img.flatten(), processed_img.flatten()], ax=ax[0])
        ax[0].set_xticks([0, 1], ['Original', 'Processed'])
        ax[0].set_title('Image Pixel Boxplot')
        ax[0].set_ylabel('Pixel Intensity')

        sns.boxplot(data=[original_label.flatten(), processed_label.flatten()], ax=ax[1])
        ax[1].set_xticks([0, 1], ['Original', 'Processed'])
        ax[1].set_title('Label Pixel Boxplot')
        ax[1].set_ylabel('Pixel Intensity')

        plt.tight_layout()
        plt.savefig('./evaluation/preprocessing/boxplot_comparison.png')
        plt.close()

        # 定义回调函数
        callbacks = [
            EarlyStopping(patience=100, verbose=2),  # 早停策略
            ReduceLROnPlateau(factor=0.5, patience=20, min_lr=0.00005, verbose=2),  # 自动降低学习率
            ModelCheckpoint('./weights/best_model.h5', verbose=2, save_best_only=True)  # 保存最佳模型
        ]

        # 训练模型
        results = self.unet.fit(
            x_train, x_label,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
            validation_split=0.1,  # 使用10%的数据作为验证集
            shuffle=True  # 打乱数据
        )

        # 绘制训练损失曲线
        loss = results.history['loss']
        val_loss = results.history['val_loss']
        metric = results.history['metric_fun']
        val_metric = results.history['val_metric_fun']
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x = np.linspace(0, len(loss), len(loss))
        plt.subplot(121)
        plt.plot(x, loss, label='loss')
        plt.plot(x, val_loss, label='val_loss')
        plt.title("Loss curve")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.subplot(122)
        plt.plot(x, metric, label='metric')
        plt.plot(x, val_metric, label='val_metric')
        plt.title("Metric curve")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Dice")
        plt.show()
        fig.savefig('./evaluation/curve.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # 测试模型
    def test(self, batch_size=1):
        os.makedirs('./evaluation/test_result', exist_ok=True)  # 创建测试结果保存目录
        self.unet.load_weights(r"weights/best_model.h5")  # 加载最佳模型权重
        x_train, x_label, y_train, y_label = self.load_data()[:4]
        test_num = y_train.shape[0]
        index, step = 0, 0
        n = 0.0
        while index < test_num:
            print('schedule: %d/%d' % (index, test_num))
            step += 1
            mask = self.unet.predict(y_train[index:index + batch_size]) > 0.1  # 预测分割掩码
            mask_true = y_label[index, :, :, 0]
            if (np.sum(mask) > 0) == (np.sum(mask_true) > 0):
                n += 1
            mask = Image.fromarray(np.uint8(mask[0, :, :, 0] * 255))  # 转换为图像格式
            mask.save('./evaluation/test_result/' + str(step) + '.png')
            mask_true = Image.fromarray(np.uint8(mask_true * 255))
            mask_true.save('./evaluation/test_result/' + str(step) + 'true.png')
            index += batch_size
        acc = n / test_num * 100
        print('the accuracy of test data is: %.2f%%' % acc)

    # 随机显示预测结果
    def test1(self, batch_size=1):
        self.unet.load_weights(r"weights/best_model.h5")
        x_train, x_label, y_train, y_label = self.load_data()[:4]
        test_num = y_train.shape[0]
        for epoch in range(5):
            rand_index = []
            while len(rand_index) < 3:
                temp = np.random.randint(0, test_num, 1)
                if np.sum(x_label[temp]) > 0:
                    rand_index.append(temp)
            rand_index = np.array(rand_index).squeeze()
            fig, ax = plt.subplots(3, 3, figsize=(18, 18))
            for i, index in enumerate(rand_index):
                mask = self.unet.predict(x_train[index:index + 1]) > 0.1
                ax[i][0].imshow(x_train[index].squeeze(), cmap='gray')
                ax[i][0].set_title('network input', fontsize=20)
                fz = 2 * np.sum(mask.squeeze() * x_label[index].squeeze())
                fm = np.sum(mask.squeeze()) + np.sum(x_label[index].squeeze())
                dice = fz / fm
                ax[i][1].imshow(mask.squeeze())
                ax[i][1].set_title('network output(%.4f)' % dice, fontsize=20)
                ax[i][2].imshow(x_label[index].squeeze())
                ax[i][2].set_title('mask label', fontsize=20)
            fig.savefig('./evaluation/show%d_%d_%d.png' % (rand_index[0], rand_index[1], rand_index[2]))
            print('finished epoch: %d' % epoch)
            plt.close()

if __name__ == '__main__':
    unet = U_Net()  # 创建U-Net模型实例
    unet.train()    # 开始训练网络并输出预处理对比图
    unet.test()     # 评价测试集
    unet.test1()    # 随机显示预测结果
