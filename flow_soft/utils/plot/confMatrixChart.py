from matplotlib import pyplot as plt

import seaborn as sns
def plot_matrix(conf_matrix,filename=''):
    # 混淆矩阵绘图
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['easy', 'optimal', 'hard'], yticklabels=['easy', 'optimal', 'hard'])
    plt.title(filename + 'Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if filename != '':
        plt.savefig(filename)
    plt.show()