import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import networkx as nx

# 사용할 큐비트 수 설정
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_lstm_gate(inputs, entangle_weights, gate_weights):
    """
    양자 회로를 통해 LSTM 게이트 연산을 구현.
    - inputs: 각 큐비트에 적용할 AngleEmbedding 입력.
    - entangle_weights: StronglyEntanglingLayers용 파라미터, shape=(1, n_qubits, 3).
    - gate_weights: RY 게이트 회전각용 파라미터, shape=(n_qubits,).
    """
    # 1. 입력 인코딩
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # 2. 강하게 얽힌 층 적용 (마지막 차원은 3이어야 함)
    qml.templates.StronglyEntanglingLayers(entangle_weights, wires=range(n_qubits))
    
    # 3. 각 큐비트에 대해 RY 게이트 회전 적용
    for i in range(n_qubits):
        qml.RY(gate_weights[i], wires=i)
    
    # 4. 추가 얽힘: 큐비트 간 CNOT 게이트
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    
    # 5. 각 큐비트의 PauliZ 기대값 측정
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits):
        super(QLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # 옵션 2: 파라미터를 분리하여 정의
        weight_shapes = {
            "entangle_weights": (1, n_qubits, 3),  # entangling layer에 사용
            "gate_weights": (n_qubits,)             # RY 게이트 회전에 사용
        }
        self.qlayer = qml.qnn.TorchLayer(quantum_lstm_gate, weight_shapes)
        
        # 입력과 이전 hidden state를 결합한 후 양자 회로의 입력으로 매핑할 선형 변환
        self.linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, x, h_prev, c_prev):
        # x: (batch_size, input_dim), h_prev: (batch_size, hidden_dim)
        combined = torch.cat((x, h_prev), dim=1)
        linear_out = self.linear(combined)
        # tanh를 거쳐 [-π, π] 범위로 변환
        q_input = torch.tanh(linear_out) * np.pi
        
        # 양자 회로 실행 (출력 shape: (batch_size, n_qubits))
        q_output = self.qlayer(q_input)
        
        # q_output의 각 열을 게이트에 대응시킴: [Input, Forget, Output, Cell Update]
        i = torch.sigmoid(q_output[:, 0]).unsqueeze(1)
        f = torch.sigmoid(q_output[:, 1]).unsqueeze(1)
        o = torch.sigmoid(q_output[:, 2]).unsqueeze(1)
        g = torch.tanh(q_output[:, 3]).unsqueeze(1)
        
        # LSTM 업데이트
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class QLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits, n_layers):
        super(QLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # n_layers개의 QLSTMCell을 생성 (첫 레이어는 입력 차원, 이후는 hidden_dim 사용)
        self.cells = nn.ModuleList([
            QLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, n_qubits)
            for i in range(n_layers)
        ])

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_dim) for _ in range(self.n_layers)]
        c = [torch.zeros(batch_size, self.hidden_dim) for _ in range(self.n_layers)]
        
        # 시퀀스의 각 타임스텝마다 QLSTMCell 순전파 실행
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = h[layer]
        
        # 마지막 레이어의 hidden state 반환
        return h[-1]

def create_graph_from_hidden_states(hidden_states, dependency_info):
    """
    hidden_states: (batch_size, hidden_dim) 형태의 QLSTM 출력
    dependency_info: 예시로 단어 간의 관계 정보를 담은 리스트 [(src, tgt), ...]
    """
    G = nx.Graph()
    num_nodes = hidden_states.shape[0]
    
    for i in range(num_nodes):
        G.add_node(i, feature=hidden_states[i].detach().numpy())
    
    for relation in dependency_info:
        src, tgt = relation
        if src < num_nodes and tgt < num_nodes:
            G.add_edge(src, tgt)
    
    return G

# 하이퍼파라미터 설정
input_dim = 10    # 입력 특징 차원
hidden_dim = 8    # LSTM hidden state 차원
n_layers = 1      # LSTM 레이어 수

# 모델 초기화
model = QLSTM(input_dim, hidden_dim, n_qubits, n_layers)

# 예시 입력 데이터
dependency_info = [(0, 1), (1, 2), (2, 3), (3, 4)]
batch_size = 5
seq_len = 7
x = torch.randn(batch_size, seq_len, input_dim)
hidden_states = torch.randn(batch_size, hidden_dim)
G = create_graph_from_hidden_states(hidden_states, dependency_info)

# 모델 실행
output = model(x)
print("Output shape:", output.shape)
print("Graph nodes with features:", G.nodes(data=True))
print("Graph edges:", G.edges())
