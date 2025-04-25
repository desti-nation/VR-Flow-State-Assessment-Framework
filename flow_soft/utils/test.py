from sklearn.feature_selection import RFE, chi2
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_friedman1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification
# from model.Machine_learning import *
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest


def RFE_demo():
    # 创建一个示例数据集
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)

    # 创建线性回归模型
    model = LinearRegression()

    # 创建RFE对象，指定模型和要选择的特征数量（这里选择3个）
    rfe = RFE(model, n_features_to_select=3)

    # 拟合RFE对象以选择特征
    fit = rfe.fit(X, y)

    # 打印特征的排名
    print("特征排名:", fit.ranking_)

    # 打印选择的特征
    print("选择的特征:", fit.support_)


def testSVM():
    # 生成一个示例数据集
    X, y = make_classification(n_samples=100, n_features=25, n_informative=3, n_redundant=2, random_state=42)

    # 创建一个SVM分类器作为基础模型
    svc = SVC(kernel="linear", random_state=42)

    # 使用RFE进行特征选择，选择5个最重要的特征
    rfe = RFE(estimator=svc, n_features_to_select=5)
    rfe.fit(X, y)

    # 输出所选特征的排名
    print("Feature ranking with SVM:", rfe.ranking_)


def testKNN():
   # 注意：KNN不直接使用coef_属性，因此我们需要使用feature_importances_属性或使用一种代理模型来评估特征重要性

    X, y = make_classification(n_samples=100, n_features=25, n_informative=3, n_redundant=2, n_classes=2,
                               random_state=42)

    # 创建一个KNN分类器作为基础模型
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

    # 使用RFE进行特征选择，选择5个最重要的特征
    rfe_knn = RFE(estimator=knn, n_features_to_select=5, step=1)

    # 使用卡方检验作为代理模型的特征选择方法
    selector = SelectKBest(score_func=chi2, k=rfe_knn.n_features_to_select)

    # 使用SelectKBest进行特征选择
    X_selected = selector.fit_transform(X, y)

    # 输出所选特征的排名
    print("Feature ranking with KNN:", selector.scores_)

import matplotlib.colors as mcolors
import numpy as np

def generate_random_colors(n):
    colors = []
    for _ in range(n):
        # 生成随机颜色（RGB）
        color = np.random.rand(3)
        # 转换为16进制格式
        hex_color = mcolors.to_hex(color)
        colors.append(hex_color)
    return colors

# 生成 n 种随机颜色

if __name__ == '__main__':
    n = 30
    random_colors = generate_random_colors(n)
    print("随机生成的 {} 种颜色（16进制）:".format(n))
    print(random_colors)
