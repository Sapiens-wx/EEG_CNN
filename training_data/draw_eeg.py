import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_data_range(csv_files=None, single_file=None):
    """
    获取数据的最大最小值，可以处理单个文件或多个文件
    """
    global_min = float('inf')
    global_max = float('-inf')
    
    if single_file:
        files_to_process = [single_file]
    else:
        files_to_process = csv_files
        
    for file in files_to_process:
        try:
            data = pd.read_csv(file)
            if len(data.columns) < 5:
                continue
                
            signal_data = data.iloc[:, 1:5]
            current_min = signal_data.values.min()
            current_max = signal_data.values.max()
            
            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)
            
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
            continue

    # 确定合适的取整单位
    def get_round_unit(value):
        abs_value = abs(value)
        if abs_value > 1000:
            return 1000
        elif abs_value > 100:
            return 100
        elif abs_value > 10:
            return 10
        else:
            return 1

    max_unit = get_round_unit(global_max)
    min_unit = get_round_unit(global_min)
    
    # 向上/下取整到最接近的单位
    max_rounded = np.ceil(global_max / max_unit) * max_unit
    min_rounded = np.floor(global_min / min_unit) * min_unit
    
    return min_rounded, max_rounded

def plot_csv_data(csv_file_path, unified_y_min=None, unified_y_max=None):
    """
    读取CSV文件并绘制四条线（y1, y2, y3, y4）随时间变化的图形。

    参数:
        csv_file_path (str): CSV文件的路径
        unified_y_min (float): 统一的Y轴最小值，如果为None则使用文件自己的范围
        unified_y_max (float): 统一的Y轴最大值，如果为None则使用文件自己的范围
    """
    try:
        # 读取CSV文件，假设第一列是时间，其余列是y1, y2, y3, y4
        data = pd.read_csv(csv_file_path)
        
        if len(data.columns) < 5:
            raise ValueError("CSV文件至少需要5列（时间 + y1, y2, y3, y4）")
          # 提取时间列和y值列
        time = data.iloc[:, 0]
        # 转换时间戳为从0开始的毫秒
        time = (time - time.iloc[0]) * 1000  # 转换为毫秒
        y1 = data.iloc[:, 1]
        y2 = data.iloc[:, 2]
        y3 = data.iloc[:, 3]
        y4 = data.iloc[:, 4]

        # 确定Y轴范围
        if unified_y_min is None or unified_y_max is None:
            y_min, y_max = get_data_range(single_file=csv_file_path)
        else:
            y_min, y_max = unified_y_min, unified_y_max

        # 获取数据类型（left/right/rest）
        data_type = "unknown"
        if "left" in csv_file_path.lower():
            data_type = "Left"
        elif "right" in csv_file_path.lower():
            data_type = "Right"
        elif "rest" in csv_file_path.lower():
            data_type = "Rest"

        # 创建图形
        plt.figure(figsize=(12, 6))
        plt.plot(time, y1, label='Channel 1', linewidth=1)
        plt.plot(time, y2, label='Channel 2', linewidth=1)
        plt.plot(time, y3, label='Channel 3', linewidth=1)
        plt.plot(time, y4, label='Channel 4', linewidth=1)        # 设置Y轴范围        plt.ylim(y_min, y_max)        
        
        # 添加垂直线标注结束时间
        end_time = time.iloc[-1]
        plt.axvline(x=end_time, color='gray', linestyle='-', alpha=0.5)  # 灰色实线，半透明
        
        # 在右边标注结束时间
        plt.text(end_time + (end_time * 0.01), (y_max + y_min) / 2, f'{end_time:.0f}ms', 
                rotation=90, verticalalignment='center')

        # 添加图例、标题和坐标轴标签
        plt.legend(loc='best')
        scale_type = "Unified scale" if unified_y_min is not None else "Individual scale"
        plt.title(f'EEG Signals - {data_type} ({os.path.basename(csv_file_path)}) - {scale_type}')
        plt.xlabel('Time')
        plt.ylabel('Signal Value')

        plt.grid(True)
        plt.tight_layout()

        # 保存图片
        output_filename = csv_file_path + ".png"
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"Generated plot for {os.path.basename(csv_file_path)}")
        
    except Exception as e:
        print(f"Error processing {csv_file_path}: {str(e)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [os.path.join(current_dir, f) for f in os.listdir(current_dir) 
                if f.endswith(".csv") and f.startswith("eeg_")]
    #csv_files=["eeg_left_20250610_162212.csv"];
    # 询问用户是否使用统一的Y轴范围
    while True:
        response = input("\nDo you want to use unified Y-axis range for all plots? (Y/n): ").lower()
        if response in ['y', 'yes', '']:  # 空字符串表示直接按回车，默认为yes
            unified_scale = True
            break
        elif response in ['n', 'no']:
            unified_scale = False
            break
        else:
            print("Please enter 'y' or 'n'")
    
    # 获取统一的Y轴范围（如果需要）
    unified_y_min = None
    unified_y_max = None
    if unified_scale:
        unified_y_min, unified_y_max = get_data_range(csv_files)
        print(f"\nUsing unified Y-axis range: [{unified_y_min}, {unified_y_max}]")
    else:
        print("\nUsing individual Y-axis range for each plot")
    
    # 处理所有符合命名规则的CSV文件
    files_processed = 0
    for file_path in csv_files:
        plot_csv_data(file_path, unified_y_min, unified_y_max)
        files_processed += 1
    
    print(f"\nTotal files processed: {files_processed}")
    print(f"Scale mode: {'Unified' if unified_scale else 'Individual'}")
    os.system("pause")
