# 标准label映射表：正式名称、缩写和所有别名
label_map = {
    "left": {
        "order" : 0,
        "abbr": "L",
        "aliases": ["left", "l"]
    },
    "right": {
        "order" : 1,
        "abbr": "R",
        "aliases": ["right", "r"]
    },
    "neutral": {
        "order" : 2,
        "abbr": "N",
        "aliases": ["neutral", "n"]
    },
    "left-to-right": {
        "order" : 3,
        "abbr": "L2R",
        "aliases": ["left-to-right", "l2r", "ltr"]
    },
    "left-to-neutral": {
        "order" : 4,
        "abbr": "L2N",
        "aliases": ["left-to-neutral", "l2n", "ltn"]
    },
    "right-to-left": {
        "order" : 5,
        "abbr": "R2L",
        "aliases": ["right-to-left", "r2l", "rtl"]
    },
    "right-to-neutral": {
        "order" : 6,
        "abbr": "R2N",
        "aliases": ["right-to-neutral", "r2n", "rtn"]
    },
    "neutral-to-left": {
        "order" : 7,
        "abbr": "N2L",
        "aliases": ["neutral-to-left", "n2l", "ntl"]
    },
    "neutral-to-right": {
        "order" : 8,
        "abbr": "N2R",
        "aliases": ["neutral-to-right", "n2r", "ntr"]
    }
}

def normalize_label(label):
    """统一标准化label字符串"""
    return label.strip().replace(' ', '-').lower()

def validate_labels(label_str, keep_order=False):
    """验证并规范化labels，返回标准labels列表和缩写列表
    
    Args:
        label_str (str): 逗号分隔的labels
        keep_order (bool): 若为True则保持输入顺序，若为False则按照order排序
    """
    # 分割并标准化输入
    labels = [normalize_label(lbl) for lbl in label_str.split(',')]
    
    # 构建别名到标准label的映射
    alias_to_label = {}
    for std_label, info in label_map.items():
        for alias in info["aliases"]:
            alias_to_label[normalize_label(alias)] = std_label
    
    # 验证并转换
    selected = []
    abbrs = []
    invalid_labels = []
    
    for lbl in labels:
        if lbl in alias_to_label:
            std_label = alias_to_label[lbl]
            selected.append(std_label)
            abbrs.append(label_map[std_label]["abbr"])
        else:
            invalid_labels.append(lbl)
    
    if not keep_order and not invalid_labels:
        # 按照order排序
        label_orders = [(label, label_map[label]["order"]) for label in selected]
        sorted_labels = [label for label, _ in sorted(label_orders, key=lambda x: x[1])]
        sorted_abbrs = [label_map[label]["abbr"] for label in sorted_labels]
        
        selected = sorted_labels
        abbrs = sorted_abbrs
    
    return selected, abbrs, invalid_labels

def get_valid_label_aliases():
    """获取所有有效的label别名列表，用于错误提示"""
    valid_aliases = []
    for info in label_map.values():
        valid_aliases.extend(info["aliases"])
    return valid_aliases

def format_valid_labels_message():
    """以格式化的方式返回有效label列表的提示信息"""
    result = "Valid labels:\n"
    
    # 按类别分组显示
    for std_label, info in sorted(label_map.items(), key=lambda x: x[1]["order"]):
        aliases = ", ".join(f"'{alias}'" for alias in info["aliases"][1:])
        if aliases:
            result += f"  - {info['aliases'][0]} (abbr: {info['abbr']}, also: {aliases})\n"
        else:
            result += f"  - {info['aliases'][0]} (abbr: {info['abbr']})\n"
    
    return result

def extract_labels_from_filename(filename):
    """
    从文件名中提取label并返回对应的label dict，按order排序
    支持录制数据、预处理数据、模型文件命名
    只处理csv、npy、keras文件
    """
    import os
    base = os.path.basename(filename)
    label_str = None
    # 录制数据: eeg_left_20250708_091151.csv
    if base.startswith('eeg_') and base.endswith('.csv'):
        parts = base.split('_')
        if len(parts) >= 3:
            label_str = parts[1]
    # 预处理数据: preprocessed_left_right_20250708_091151.npy
    elif base.startswith('preprocessed_') and base.endswith('.npy'):
        label_str = base[len('preprocessed_'):].split('_')[0]
        # 支持多个label
        if '_' in base[len('preprocessed_'):]:
            label_str = base[len('preprocessed_'):].split('_')[0]
            # 如果有多个label（如 left_right），用下划线分割
            label_str = label_str.replace('-', ',').replace('_', ',')
    # 模型文件: model_left_right_20250708_091151.keras
    elif base.startswith('model_') and base.endswith('.keras'):
        label_str = base[len('model_'):].split('_')[0]
        if '_' in base[len('model_'):]:
            label_str = base[len('model_'):].split('_')[0]
            label_str = label_str.replace('-', ',').replace('_', ',')
    if not label_str:
        return []
    # 分割并标准化
    labels = [normalize_label(lbl) for lbl in label_str.split(',')]
    # 构建别名到标准label的映射
    alias_to_label = {}
    for std_label, info in label_map.items():
        for alias in info["aliases"]:
            alias_to_label[normalize_label(alias)] = std_label
    # 收集label dict
    label_dicts = []
    for lbl in labels:
        if lbl in alias_to_label:
            std_label = alias_to_label[lbl]
            label_dicts.append({std_label: label_map[std_label]})
    # 按order排序
    label_dicts_sorted = sorted(label_dicts, key=lambda d: list(d.values())[0]["order"])
    return label_dicts_sorted

def get_label_count_from_filename(filename):
    """
    获取文件名中的label数量，服用extract_labels_from_filename
    """
    label_dicts = extract_labels_from_filename(filename)
    return len(label_dicts)
