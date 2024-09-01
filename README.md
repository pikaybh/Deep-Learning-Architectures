# Deep Learning Architectures

**TOC**

- [Introduction](#introduction)
- [1. Neural Net; NN](#1-neural-net-nn)
- [2. Convolutional Neural Network; CNN](#)
- [3. Recurrent Neural Network; RNN](#)
- [4. Long Short-Term Memory; LSTM](#)
- [5. Transformer](#)
- [6. Visual Transformer](#)

## Introduction

개인적으로 DL 모델들을 공부하기 위해 Architecture 별로 정리해둔 Repo입니다.

### Numpy Module

Numpy 설명

### PyTorch Framework

PyTorch에는 중요한 **세 가지 명시적 추상화 수준(Three Explicit Levels of Abstraction)**이 있다.

- **Tensor**
    - Tensor는 Interative ndarray이다. 
    - Tensor Flow의 Numpy array와 같은 역할을 함.
    - 한 가지 다른 점은 GPU에서도 동작한다는 것

- **Variable**
    - 계산 그래프(Computational Graph)를 구성하는 노드(Node)
    - 데이터와 기울기(gradient)를 저장함

- **Module**
    - Module 객체를 이용해서 Neural Net Layer를 구성할 수 있음
    - state와 학습 가능한 weight를 저장할 수도 있음

#### PyTorch vs TensorFlow

| PyTorch | TensorFlow |
| ------- | ---------- |
| Tensor | Numpy array |
| Variable | Tensor, Variable, Placeholder |
| Module | Tf.layers, TFSlim, TFLearn, etc. |

#### PyTorch Implementation

| TOC |
| --- |
| 2-Layer Net with PyTorch Tensor |
| 2-Layer Net with PyTorch Autograd |
| Define New Autograd Function |
| 2-Layer Net with PyTorch nn |
| PyTorch nn: Define new Modules |

대부분 다음과 같은 학습 흐름을 가지고 있음.

> Tensor생성 -> Forward -> Backward -> Weight Update

##### 2-Layer Net with PyTorch Tensor

```python
import torch
```

**Cuda GPU Computation** `Torch.cuda`를 통한 GPU연산

```python
dtype = torch.cuda.FloatTensor
```

**Step 1.** Data와 Weight을 random tensor로 생성

```python
N, D_in, H, D_out = 64, 1_000, 100, 10
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)
```

**Step 2.** Forward pass; Prediction, loss 값 계산

```python
learnig_rate = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.relu.mm(w2)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()
    ...
```

**Step 3.** Backward pass; gradient 계산

```python
    ...
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    ...
```

**Step 4.** Weight 업데이트 (Gradient Descent Steps)

```python
    ...
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```
##### 2-Layer Net with PyTorch Autograd

출처: https://deepinsight.tistory.com/101 [Steve-Lee's Deep Insight:티스토리]

## 1. Neural Net; NN

NN은 모델을 정의하고 미분하는데 `autograd`를 이용한다.
`nn.Module`은 층(Layer)과 output을 반환하는 `forward(input)` 메서드를 포함하고 있다.

<figure>
    <img src="src\LeNet-5 Architecture as Published in the original paper.png" alt="LeNet-5 Architecture as Published in the original paper">
    <figcaption>LeNet-5 Architecture as Published in the original paper</figcaption>
</figure>

**LeNet**은 간단한 *순전파 네트워크(Feed-forward network)* 이다.
입력(input)을 받아 여러 계층에 차례로 전달한 후, 최종 출력(output)을 제공한다.
신경망의 일반적인 학습과정은 다음과 같다:

1. 학습 가능한 매개변수(parameters)/가중치(weight)를 갖는 **신경망을 정의**함.
2. **데이터셋(dataset)입력**을 반복
3. 입력을 신경망에서 전파(**process**)
4. 손실(**loss**: 출력이 정답으로부터 얼마나 떨어져 있는지)을 계산
5. 변화도(**gradient**)를 신경망의 매개변수들에 역으로 전파; **Backpropagation**
6. 신경망의 가중치를 갱신함. *일반적으로 다음과 같은 간단한 규칙을 사용한다.*

$`업데이트된 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gradient)`$


$`W_1 = W_0 - \eta * \frac{df}{dx}`$

### 1.1. 신경망 정의하기

