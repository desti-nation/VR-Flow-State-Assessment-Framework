import csv
import pickle
from scipy.stats import sem
import numpy as np
from matplotlib import pyplot as plt
from CommonParam import *
"""
为主观得分（包括问卷和三维打分结果）绘制带误差线的柱状图

基本不可为外部文件调用，可以看作是一个独立的绘图单元（指定文件来源和读取方式，以及后续的绘图参数）
"""
error_kw = {'capsize': 2, 'elinewidth': 0.5, 'markeredgewidth': 0.5} # 误差线设置
bar_width = 3
colors = CommonParam.colors
y1_axis_color = 'blue'
y2_axis_color = 'red'
def readpkl():
    with open('plotData.pkl', 'rb') as file:
        plotData = pickle.load(file)
    return plotData

def readcsv():
    y1 = []
    y2 = []
    y3 = []
    item_list = []
    with open('subject.csv','r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            item_name = row['item']
            if item_name in item_list:
                y1[len(item_list)-1].append(float(row['easy']))
                y2[len(item_list)-1].append(float(row['optimal']))
                y3[len(item_list)-1].append(float(row['hard']))
            else:
                item_list.append(item_name)
                y1.append([])
                y2.append([])
                y3.append([])
    plotData = {
        'y1':y1,
        'y2':y2,
        'y3':y3,
        'item_list':item_list
    }
    with open('plotData.pkl','wb') as file:
        pickle.dump(plotData,file)
    return plotData

def barPlot(plotData):
    item_list = plotData['item_list']
    item_n = len(item_list)
    n_3fs = 4
    y1,y2,y3 = plotData['y1'],plotData['y2'],plotData['y3']
    y = [y1,y2,y3]
    x1 = [1 + 11 * i for i in range(item_n)]
    x2 = [4 + 11 * i for i in range(item_n)]
    x3 = [7 + 11 * i for i in range(item_n)]
    x = [x1,x2,x3]

    split_line = x1[item_n - n_3fs] - 2.5 # 绘制分割线

    fig,ax1 = plt.subplots(figsize=(16, 4), dpi=100)

    ax1.yaxis.grid(True, linestyle='--', color='grey', alpha=0.1, zorder=-2)
    for i in range(3):
        mean_value = [ np.mean(np.array(y[i][_])) for _ in range(item_n-n_3fs)]
        error_value = [ sem(np.array(y[i][_])) for _ in range(item_n -n_3fs)]
        ax1.bar(x[i][:item_n - n_3fs],mean_value,bar_width,color=colors[i],
                yerr=error_value,error_kw=error_kw, label=['easy','optimal','hard'][i],zorder=5)
    ax1.set_xticks(x2)
    ax1.set_xticklabels(item_list)
    ax1.set_ylim(0,5.5)
    ax1.set_xlim(-3,x3[-1]+4)
    ax1.set_yticks([0,1,2,3,4,5])
    ax1.tick_params(axis='x',direction='in')
    ax1.tick_params(axis='y',direction='in',labelcolor=y1_axis_color)


    ax2 = ax1.twinx()
    #ax2.yaxis.grid(True, linestyle='--', color='grey', alpha=0.1, zorder=-2)
    for i in range(3):
        mean_value = [np.mean(np.array(y[i][_])) for _ in range(item_n - n_3fs,item_n)]
        error_value = [sem(np.array(y[i][_])) for _ in range(item_n - n_3fs,item_n)]
        ax2.bar(x[i][item_n - n_3fs:],mean_value,bar_width,color=colors[i],
                yerr=error_value,error_kw=error_kw,  label=['easy','optimal','hard'][i],zorder=5)
    #ax2.set_ylabel('Dataset 2 values', color='red')
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.tick_params(axis='x',direction='in')
    ax2.tick_params(axis='y',direction='in',labelcolor=y2_axis_color)

    plt.axvline(x=split_line, linestyle='--', color='black', lw=1)

    # 设置图例
    #ax1.legend(loc='upper left')
    #ax2.legend(loc='upper right')

    #plt.title('')
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.savefig('barplot.png', dpi=500)
    plt.show()

if __name__ == '__main__':
    plotData = readpkl()
    #plotData = readcsv()
    barPlot(plotData)