import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, jaccard_score, accuracy_score
import os

# 定义所有可能的舌头类型，已移除 '正常'
all_classes = ["齿痕舌", "裂纹舌", "瘀点舌", "剥苔舌", "腻苔舌", "薄苔舌", "灰黑舌"]

def parse_labels(text, classes):
    """
    解析预测标签文本，提取所有舌象类型，忽略 '正常' 标签。

    参数:
        text (str): 包含预测标签的文本。
        classes (set): 有效的舌象类别集合。

    返回:
        list: 预测舌象标签列表。
    """
    labels = []
    unknown_labels = set()

    if not text:
        return labels  # 返回空列表

    # 按行分割
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('**') and line.endswith('**'):
            # 提取舌象名称
            label = line.strip('*').strip()
            if label in classes:
                labels.append(label)
            elif label != '正常':  # 只忽略 '正常'，记录其他未知标签
                unknown_labels.add(label)

    if unknown_labels:
        print(f"发现未知标签: {unknown_labels}")

    return labels

def load_true_labels_from_excel(excel_file, classes):
    """
    从 Excel 文件中加载真实标签。

    参数:
        excel_file (str): 真实标签的 Excel 文件路径。
        classes (list): 所有类别列表。

    返回:
        ndarray: 真实标签的二进制编码矩阵。
    """
    df = pd.read_excel(excel_file)
    
    # 确保所有类别列存在
    for cls in classes:
        if cls not in df.columns:
            raise ValueError(f"Excel 文件中缺少类别列: {cls}")

    # 提取类别列
    Y_true = df[classes].values

    return Y_true

def load_pred_labels_from_json(json_file, classes):
    """
    从 JSON 文件中加载预测标签。

    参数:
        json_file (str): 预测标签的 JSON 文件路径。
        classes (list): 所有类别列表。

    返回:
        list: 预测标签的列表，每个元素是一个标签列表。
    """
    pred_labels = []

    with open(json_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON 解码错误在第 {line_num} 行: {e}")
                continue  # 跳过有问题的行

            predict_text = data.get('predict', '')
            labels = parse_labels(predict_text, set(classes))
            pred_labels.append(labels)

    return pred_labels

def compute_per_class_accuracy(Y_true, Y_pred, classes):
    """
    计算每个类别的准确率。

    参数:
        Y_true (ndarray): 真实标签的二进制编码。
        Y_pred (ndarray): 预测标签的二进制编码。
        classes (list): 类别名称列表。

    返回:
        dict: 每个类别的准确率。
    """
    per_class_accuracy = {}
    for i, cls in enumerate(classes):
        TP = ((Y_true[:, i] == 1) & (Y_pred[:, i] == 1)).sum()
        TN = ((Y_true[:, i] == 0) & (Y_pred[:, i] == 0)).sum()
        FP = ((Y_true[:, i] == 0) & (Y_pred[:, i] == 1)).sum()
        FN = ((Y_true[:, i] == 1) & (Y_pred[:, i] == 0)).sum()

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        per_class_accuracy[cls] = accuracy

    return per_class_accuracy

def evaluate(Y_true, Y_pred, classes, output_dir):
    """
    评估多标签分类的性能，并将评估结果保存到文件中。

    参数:
        Y_true (ndarray): 真实标签的二进制编码。
        Y_pred (ndarray): 预测标签的二进制编码。
        classes (list): 所有类别。
        output_dir (str): 输出文件的目录路径。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存真实标签为 Excel 文件
    true_df = pd.DataFrame(Y_true, columns=classes)
    true_excel_file = os.path.join(output_dir, "true_labels.xlsx")
    true_df.to_excel(true_excel_file, index=False)
    print(f"真实标签的二进制编码已保存到 {true_excel_file}")

    # 保存预测标签为 Excel 文件
    pred_df = pd.DataFrame(Y_pred, columns=classes)
    pred_excel_file = os.path.join(output_dir, "pred_labels.xlsx")
    pred_df.to_excel(pred_excel_file, index=False)
    print(f"预测标签的二进制编码已保存到 {pred_excel_file}")

    # 计算评估指标
    report = classification_report(Y_true, Y_pred, target_names=classes, digits=4)
    hamming = hamming_loss(Y_true, Y_pred)
    jaccard = jaccard_score(Y_true, Y_pred, average='samples')
    subset_acc = accuracy_score(Y_true, Y_pred)

    # 计算每个类别的准确率
    per_class_acc = compute_per_class_accuracy(Y_true, Y_pred, classes)

    # 组织评估结果
    evaluation_results = f"""
分类报告:
{report}

汉明损失 (Hamming Loss): {hamming:.4f}
Jaccard 相似度 (Jaccard Score): {jaccard:.4f}
子集准确率 (Subset Accuracy): {subset_acc:.4f}

每个类别的准确率:
"""
    for cls, acc in per_class_acc.items():
        evaluation_results += f"{cls}: {acc:.4f}\n"

    # 保存评估报告
    report_file = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(evaluation_results)

    print(f"评估报告已保存到 {report_file}")

def main():
    # 文件路径设置/media/ubuntu/shr/SHR-LLaMA-Factory/saves/llava-next-mistral-7b/12_27/test/generated_predictions.jsonl
    true_labels_excel = './test_output.xlsx'  # 替换为您的真实标签 Excel 文件路径
    pred_labels_json = './saves/paligemma2-3b-pt-448/1_9/test/generated_predictions.jsonl'  # 替换为您的预测标签 JSON 文件路径
    output_directory = './saves/paligemma2-3b-pt-448/1_9/test/'  # 请确保该目录存在或将其创建

    # 读取真实标签
    Y_true = load_true_labels_from_excel(true_labels_excel, all_classes)
    print(f"成功加载真实标签，样本数量: {Y_true.shape[0]}")

    # 读取预测标签
    pred_labels = load_pred_labels_from_json(pred_labels_json, all_classes)
    print(f"成功加载预测标签，样本数量: {len(pred_labels)}")

    # 确认样本数量一致
    if Y_true.shape[0] != len(pred_labels):
        raise ValueError(f"真实标签样本数量 ({Y_true.shape[0]}) 与预测标签样本数量 ({len(pred_labels)}) 不一致。请检查数据文件。")

    # 将预测标签转换为二进制编码
    mlb = MultiLabelBinarizer(classes=all_classes)
    Y_pred = mlb.fit_transform(pred_labels)

    # 评估模型
    evaluate(Y_true, Y_pred, all_classes, output_directory)

if __name__ == "__main__":
    main()
