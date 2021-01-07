# 此程式碼複製自 https://zhuanlan.zhihu.com/p/114414797
import numpy


def cer(r: list, h: list):
    """
    Calculation of CER with Levenshtein distance.
    """
    # initialisation
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / float(len(r))


if __name__ == "__main__":
    reference = list('abcdef')
    hypothesis = list('cdefgh')
    r = [x for x in reference]
    h = [x for x in hypothesis]
    print(cer(r, h))

    reference = list('今天天氣好好呦')
    hypothesis = list('今天天氣很好耶')
    r = [x for x in reference]
    h = [x for x in hypothesis]
    print(cer(r, h))
