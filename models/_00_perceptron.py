import numpy as np
from typing import Union

# Perceptron
def _perceptron(w: np.ndarray, x: np.ndarray, b: Union[int, float]) -> np.ndarray:
    '''
    $\sum{w \cdot x} + b$

    :param w: an array of weights
    :type w: np.ndarray
    :param x: an array of inputs
    :type x: np.ndarray
    :param b: bias
    :type b: Union[int, float]
    :return: y
    :rtype: np.ndarray
    '''
    return np.sum(w * x) + b

# Single-layer Perceptrons
def AND(x1: int, x2: int, w1: Union[str, float] = .5, w2: Union[str, float] = .5, b: Union[str, float] = -.7) -> int:
    '''
    An example of AND Gate

    :param x1: input (0 or 1)
    :type x1: int
    :param x2: input (0 or 1)
    :type x2: int
    :param w1: weight for x1
    :type w1: Union[int, float]
    :param w2: weight for x2
    :type w2: Union[int, float]
    :param b: bias
    :type b: Union[int, float]
    :return: output (0 or 1)
    :rtype: int
    '''
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    return 1 if 0 < _perceptron(w, x, b) else 0

def NAND(x1: int, x2: int, w1: Union[str, float] = -.5, w2: Union[str, float] = -.5, b: Union[str, float] = .7) -> int:
    '''
    An example of NAND Gate

    :param x1: input (0 or 1)
    :type x1: int
    :param x2: input (0 or 1)
    :type x2: int
    :param w1: weight for x1
    :type w1: Union[int, float]
    :param w2: weight for x2
    :type w2: Union[int, float]
    :param b: bias
    :type b: Union[int, float]
    :return: output (0 or 1)
    :rtype: int
    '''
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    return 1 if 0 < _perceptron(w, x, b) else 0

def OR(x1: int, x2: int, w1: Union[str, float] = .5, w2: Union[str, float] = .5, b: Union[str, float] = -.2) -> int:
    '''
    An example of OR Gate

    :param x1: input (0 or 1)
    :type x1: int
    :param x2: input (0 or 1)
    :type x2: int
    :param w1: weight for x1
    :type w1: Union[int, float]
    :param w2: weight for x2
    :type w2: Union[int, float]
    :param b: bias
    :type b: Union[int, float]
    :return: output (0 or 1)
    :rtype: int
    '''
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    return 1 if 0 < _perceptron(w, x, b) else 0

# Multi-layer Perceptron
def XOR(x1: Union[int, float], x2: Union[int, float]) -> int:
    '''
    An example of XOR Gate

    :param x1: input (0 or 1)
    :type x1: int
    :param x2: input (0 or 1)
    :type x2: int
    :return: output (0 or 1)
    :rtype: int
    '''
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def main() -> None:
    print("AND")
    print(AND(0, 0))  # 0
    print(AND(0, 1))  # 0
    print(AND(1, 0))  # 0
    print(AND(1, 1))  # 1

    print("NAND")
    print(NAND(0, 0))  # 1
    print(NAND(0, 1))  # 1
    print(NAND(1, 0))  # 1
    print(NAND(1, 1))  # 0

    print("OR")
    print(OR(0, 0))  # 0
    print(OR(0, 1))  # 1
    print(OR(1, 0))  # 1
    print(OR(1, 1))  # 1

    print("XOR")
    print(XOR(0, 0))  # 0
    print(XOR(0, 1))  # 1
    print(XOR(1, 0))  # 1
    print(XOR(1, 1))  # 0

# Main
if __name__ == '__main__':
    main()

# XOR은 2층 퍼셉트론이다.
# 2층 퍼셉트론(=비선형 시그노이드 함수)를 활성화 함수로 사용하면
# 임의의 함수를 표현할 수 있다는 사실이 증명되어 있다.