import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class QGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_qubits):
        super(QGNNLayer, self).__init__()
        self.n_qubits = n_qubits
        self.qnode = qml.qnode(dev, interface="torch")(self.quantum_circuit)

        # 선형 변환
        self.linear = nn.Linear(in_dim, n_qubits)

    def quantum_circuit(self, inputs):
        qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
        qml.templates.StronglyEntanglingLayers(inputs, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        q_input = torch.tanh(self.linear(x))
        q_output = self.qnode(q_input)
        return q_output
