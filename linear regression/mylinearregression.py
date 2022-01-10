import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

class MyLinearRegression():
    def __init__(self):
        self.w = []
        pass

    def propagate(self, theta_0, theta_1, miliage, price, step_size, m):
        # print(np.sum(theta_0 + miliage * theta_1 - price))
        # print(np.dot(miliage, theta_1.T))
        # print (np.sum(theta_0 + miliage * theta_1 - price) * miliage)
        theta_0 = theta_0 - (step_size / m) * np.sum(theta_0 + np.dot(miliage, theta_1.T) - price)
        theta_1 = theta_1 - ((step_size / m) * (np.sum(theta_0 + np.dot(miliage, theta_1.T) - price)) * miliage)
        self.w.append((theta_0, theta_1))
        # print(theta_0, theta_1)
        return theta_0, theta_1

    def training_part(self, theta_0, theta_1, steps_count, step_size):
        data_train = pd.read_csv("data.csv")
        plt.imshow(data_train.km, data_train.price)
        m = data_train.shape[0]
        theta_1 = np.zeros((m, 1))
        price = data_train[["price"]].to_numpy() # нужно убрать заголовочную строку
        milliage = data_train[["km"]].to_numpy() # и тут
        price, milliage = self.normalize_datas(price, milliage)
        # print(miliage.shape)
        # print(price)
        # print(price.shape)
        # print(theta_1.shape)

        for i in range(steps_count):
            theta_0, theta_1 = self.propagate(theta_0, theta_1, milliage, price, step_size, m)

        # print(theta_0, theta_1)

    def predicting_part(self):
        theta_0 = 0
        theta_1 = 0
        theta_0, theta_1 = self.training_part(theta_0, theta_1, 100, 10)

        data_test = pd.read_csv("data.csv")

        miliage = data_test["km"]
        estimate_price = theta_0 + miliage * theta_1
        # print(miliage, estimate_price)

    def get_weights(self):
        return self.w

    def normalize_datas(self, price, miliage):
        return price, miliage