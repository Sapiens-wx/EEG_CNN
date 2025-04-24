import os;
import matplotlib.pyplot as plt
import pandas as pd

def plot_csv_data(csv_file_path):
    """
    读取CSV文件并绘制四条线（y1, y2, y3, y4）随时间变化的图形。

    参数:
        csv_file_path (str): CSV文件的路径。
    """
    # 读取CSV文件，假设第一列是时间，其余列是y1, y2, y3, y4
    data = pd.read_csv(csv_file_path)
    
    # 检查数据列数是否符合要求（至少5列：时间 + y1, y2, y3, y4）
    if len(data.columns) < 5:
        raise ValueError("CSV文件至少需要5列（时间 + y1, y2, y3, y4）")
    
    # 提取时间列（第一列）和y值列（第二列到第五列）
    time = data.iloc[:, 0]  # 第一列是时间
    y1 = data.iloc[:, 1]    # 第二列是y1
    y2 = data.iloc[:, 2]    # 第三列是y2
    y3 = data.iloc[:, 3]    # 第四列是y3
    y4 = data.iloc[:, 4]    # 第五列是y4

    # 创建图形
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 绘制四条线
    plt.plot(time, y1, label='y1', linewidth=2)
    plt.plot(time, y2, label='y2', linewidth=2)
    plt.plot(time, y3, label='y3', linewidth=2)
    plt.plot(time, y4, label='y4', linewidth=2)

    # 添加图例、标题和坐标轴标签
    plt.legend(loc='best')
    plt.title('四条线随时间变化')
    plt.xlabel('时间')
    plt.ylabel('值')

    # 显示图形
    plt.grid(True)  # 添加网格
    plt.tight_layout()  # 调整布局

    # save
    plt.savefig(csv_file_path+".png");
    plt.close();

if __name__ == "__main__":
    plot_csv_data("test.csv");
    os.system("pause");
