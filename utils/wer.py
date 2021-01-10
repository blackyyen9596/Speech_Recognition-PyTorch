# 此程式碼複製自 https://blog.csdn.net/baobao3456810/article/details/107381052
# 並經過本人更改為wer之函式
import numpy as np


def levenshtein_distance(hypothesis: list, reference: list):
    """编辑距离
    计算两个序列的levenshtein distance，可用于计算 WER/CER
    参考资料：https://martin-thoma.com/word-error-rate-calculation/

    C: correct
    W: wrong
    I: insert
    D: delete
    S: substitution

    :param hypothesis: 预测序列
    :param reference: 标注序列
    :return: 1: 错误操作，所需要的 S，D，I 操作的次数;
             2: ref 与 hyp 的所有对齐下标
             3: 返回 C、W、S、D、I 各自的数量
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    # 记录所有的操作，0-equal；1-insertion；2-deletion；3-substitution
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    # 生成 cost 矩阵和 operation矩阵，i:外层hyp，j:内层ref
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i - 1] == reference[j - 1]:
                cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
            else:
                substitution = cost_matrix[i - 1][j - 1] + 1
                insertion = cost_matrix[i - 1][j] + 1
                deletion = cost_matrix[i][j - 1] + 1

                compare_val = [insertion, deletion, substitution]  # 优先级
                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []  # 保存 hyp与ref 中所有对齐的元素下标
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_hyp, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:  # correct
            if i - 1 >= 0 and j - 1 >= 0:
                match_idx.append((j - 1, i - 1))
                nb_map['C'] += 1

            # 出边界后，这里仍然使用，应为第一行与第一列必然是全零的
            i -= 1
            j -= 1
        elif ops_matrix[i_idx][j_idx] == 1:  # insert
            i -= 1
            nb_map['I'] += 1
        elif ops_matrix[i_idx][j_idx] == 2:  # delete
            j -= 1
            nb_map['D'] += 1
        elif ops_matrix[i_idx][j_idx] == 3:  # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1

        # 出边界处理
        if i < 0 and j >= 0:
            nb_map['D'] += 1
        elif j < 0 and i >= 0:
            nb_map['I'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt
    wer = (nb_map['S'] + nb_map['D'] +
           nb_map['I']) / (nb_map['S'] + nb_map['D'] + nb_map['C'])
    # print("ref: %s" % " ".join(reference))
    # print("hyp: %s" % " ".join(hypothesis))
    # print(nb_map)
    # print("match_idx: %s" % str(match_idx))
    # print()
    # return wrong_cnt, match_idx, nb_map
    return wer


def test():
    reference = list('abcdef')
    hypothesis = list('cdefgh')
    wer = levenshtein_distance(reference=reference, hypothesis=hypothesis)
    print(wer)

    reference = list('今天天氣很好呦')
    hypothesis = list('今天天氣很好耶')
    wer = levenshtein_distance(reference=reference, hypothesis=hypothesis)
    print(wer)


if __name__ == '__main__':
    test()