import torch.nn as nn
import pennylane as qml
import torch
from pennylane import numpy as np

n_qubits = 4  # 사용할 큐비트 수 (LSTM 게이트 개수에 맞춤)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_lstm_gate(inputs, weights):
    """QLSTM 게이트 연산을 수행하는 양자 회로"""
    
    # 입력 데이터를 양자 상태로 변환
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    # 가중치 shape 명확히 설정
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # QLSTM 게이트 연산 수행 (Forget, Input, Output, Cell Update)
    qml.RY(weights[0], wires=0)  # Forget Gate
    qml.RY(weights[1], wires=1)  # Input Gate
    qml.RY(weights[2], wires=2)  # Output Gate
    qml.RY(weights[3], wires=3)  # Cell Update Gate

    # 큐비트 간 얽힘 추가
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])

    # 측정 수행
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers=3):
        super(QLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # ✅ 가중치 shape 수정
        weight_shapes = {"weights": (n_layers, n_qubits, 4)}
        self.qlayer = qml.qnn.TorchLayer(quantum_lstm_gate, weight_shapes)

        # 선형 변환 정의
        self.linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, x, h_prev, c_prev):
        """QLSTM 셀의 순전파 연산"""
        combined = torch.cat((x, h_prev), dim=1)
        linear_out = self.linear(combined)
        q_input = torch.tanh(linear_out) * np.pi  # [-π, π] 범위로 변환

        # ✅ weight shape이 맞춰진 양자 회로 실행
        q_output = self.qlayer(q_input)

        # LSTM 게이트 계산
        i = torch.sigmoid(q_output[0])  # Input Gate
        f = torch.sigmoid(q_output[1])  # Forget Gate
        o = torch.sigmoid(q_output[2])  # Output Gate
        g = torch.tanh(q_output[3])     # Cell Update

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class QLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers):
        super(QLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # QLSTM 셀 생성
        self.cells = nn.ModuleList([
            QLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, n_qubits) 
            for i in range(n_layers)
        ])

    def forward(self, x):
        """QLSTM 전체 모델의 순전파 연산"""
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_dim) for _ in range(self.n_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim) for _ in range(self.n_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = h[layer]

        return h[-1]  # 최종 출력 Hidden State 반환
# 하이퍼파라미터 설정
input_dim = 10  # 입력 특징의 차원
hidden_dim = 8  # LSTM의 hidden state 차원
n_qubits = 4    # 사용할 큐비트 수
n_layers = 1    # LSTM 레이어 수

# 모델 초기화
model = QLSTM(input_dim, hidden_dim, n_qubits, n_layers)

# 예시 입력 데이터
batch_size = 5
seq_len = 7
x = torch.randn(batch_size, seq_len, input_dim)

# 모델 실행
output = model(x)
print(output.shape)  # 예상 출력: torch.Size([5, 8])
