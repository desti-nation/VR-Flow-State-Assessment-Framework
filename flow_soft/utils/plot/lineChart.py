
from matplotlib import pyplot as plt
from .CommonParam import *

def linePlot(x,y,n=1,title=None,label=None,path=None,color=CommonParam.line, figsize=(15, 3)):
    return
    plt.figure(figsize=figsize)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.grid(True, axis='y', linestyle='--', color='grey', alpha=0.1)

    if n == 1:
        plt.plot(x,y,color=color,linewidth=1)
        x_range= x[-1] - x[0]
        plt.xlim(-0.005*x_range,x[-1]+0.005*x_range)
    else:
        for i in range(n):
            plt.plot(x[i],y[i],color=color,linewidth=1)

    if label is not None:
        plt.legend([label], loc='upper right')  # 图例
    if title is not None:
        plt.title(title)
    if path is not None:
        plt.savefig(path, dpi=400,bbox_inches='tight')

    plt.subplots_adjust(left=0.08, right=0.98,top=0.95,bottom=0.1)
    plt.show()


# 分阶段绘图可以来一张

    """
    def plot_chip_data(self, item_name=None):
        item_list = self.data.keys() if item_name is None else {item_name}

        for item in item_list:
            try:
                MyPlot.linechart_with_epoch(data=self.chip_data[item],
                                            color=Color.getColor(self.father.latinIdx[self.player]),
                                            title='player-' + str(self.player) + '-(chip)' + item,
                                            path='D:\\signal\\flow_project\\FeatureRec', figsize=(12, 6))
            except Exception as e:
                self.my_print(f'{e}')
                continue

    """