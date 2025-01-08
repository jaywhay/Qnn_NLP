import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, AutoTokenizer

# -- (전처리) 동일 폴더 내의 'data_preprocessing.py' 에서 불러온다고 가정
# from data_preprocessing import train_dataset, val_dataset, tokenizer
from NewData import train_dataset, val_dataset, tokenizer

###################################
# Qiskit 관련 import
###################################
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Depth, Optimize1qGates, CommutativeCancellation
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator

###################################
# 1. Autoencoder 정의
###################################
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # 단순화된 Encoder만 (Decoder는 양자 디코더로 대체할 것이라 가정)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # 원래 Autoencoder라면 Decoder도 들어있어야 하지만,
        # 여기서는 "양자 디코딩" 부분으로 대체한다고 가정.

    def forward(self, x):
        return self.encoder(x)

###################################
# 2. 양자 회로 및 VQE 레이어
###################################
class QuantumCircuitWithParams(nn.Module):
    """
    PyTorch Module처럼 동작하며, 학습 가능한 파라미터를 포함한 양자 회로.
    """
    def __init__(self, num_qubits):
        super(QuantumCircuitWithParams, self).__init__()
        self.num_qubits = num_qubits
        # 예시: 각 큐비트마다 RZ 게이트의 파라미터 하나씩
        self.trainable_params = nn.Parameter(torch.rand(num_qubits))

    def forward(self, data_vector):
        """
        data_vector: (num_qubits,) 크기의 텐서라고 가정
        """
        qc = QuantumCircuit(self.num_qubits)
        # data_vector와 trainable_params를 함께 사용해 회전 게이트 적용
        for i in range(self.num_qubits):
            param = float(self.trainable_params[i].item())
            val = float(data_vector[i].item())
            qc.ry(val * param, i)  # 예: RY 게이트
            qc.rz(val * param / 2, i)  # 예: RZ 게이트

        # 간단히 모든 큐비트에 걸쳐 CNOT 반복
        for i in range(self.num_qubits - 1):
            qc.cx(i, i+1)

        return qc

def generate_sparse_hamiltonian(data_vector):
    """
    data_vector를 기반으로 SparsePauliOp 생성.
    상관관계가 높은 특징만 골라서 Sparse하게 만든다고 가정.
    """
    data_vector = data_vector.detach().cpu().numpy()  # 넘파이 변환
    num_qubits = len(data_vector)
    # 간단하게 Z 연산자를 데이터 값 가중치로 곱해준다 가정
    # 실제로는 상관관계가 높은 항만 남겨서 Sparse하게 만드는 로직을 구현해야 함
    pauli_list = []
    for i, coeff in enumerate(data_vector):
        # 예: 'Z'가 num_qubits개인 문자열에서 i번째만 Z, 나머지는 I가 되도록
        # 너무 간단하게 구성된 예시
        label = ['I'] * num_qubits
        label[i] = 'Z'
        label = "".join(label)
        pauli_list.append((label, float(coeff)))

    return SparsePauliOp.from_list(pauli_list)


class VQEQuantumLayer(nn.Module):
    """
    PyTorch forward에서,
    1) 데이터 기반으로 해밀토니안 생성
    2) parameterized quantum circuit 생성
    3) VQE 실행 후 에너지 값을 loss로 반환
    """
    def __init__(self, circuit_module, estimator=None):
        super(VQEQuantumLayer, self).__init__()
        self.circuit_module = circuit_module
        # Qiskit primitives
        self.estimator = estimator if estimator is not None else Estimator()

    def forward(self, reduced_embeddings):
        """
        reduced_embeddings: (batch_size, latent_dim) 형태
        배치 병렬 처리를 위해 각 샘플별로 독립적인 VQE를 수행한 뒤 평균
        """
        quantum_losses = []
        batch_size = reduced_embeddings.size(0)
        for i in range(batch_size):
            data_vector = reduced_embeddings[i]  # (latent_dim,)
            # 1) Sparse Hamiltonian 생성
            hamiltonian = generate_sparse_hamiltonian(data_vector)

            # 2) 파라미터화된 양자 회로 생성
            qc = self.circuit_module(data_vector)
            # VQE를 직접 여기서 돌리는 대신, 
            # Qiskit Primitives의 Estimator.run() 사용해 기대값을 측정
            # -> VQE와 유사한 효과

            # estimator.run()은 리스트 형태 인자를 받음
            try:
                job = self.estimator.run(
                    circuits=[qc],
                    observables=[hamiltonian],
                )
                result = job.result()
                energy = result.values[0]
            except Exception as e:
                print(f"Estimator error: {e}")
                # 실패 시 0으로 처리(또는 다른 방식 예외처리)
                energy = 0.0

            quantum_losses.append(energy)

        if len(quantum_losses) == 0:
            return torch.tensor(0.0)
        else:
            return torch.tensor(quantum_losses, dtype=torch.float32).mean()

###################################
# 3. 최종 Hybrid Model
###################################
class HybridModel(nn.Module):
    def __init__(self, encoder, autoencoder, quantum_layer, vocab_size):
        super(HybridModel, self).__init__()
        self.encoder = encoder         # BERT 모델
        self.autoencoder = autoencoder # 차원 축소용
        self.quantum_layer = quantum_layer
        self.output_layer = nn.Linear(autoencoder.encoder[-1].out_features, vocab_size)

    def forward(self, input_ids, attention_mask):
        # 1) BERT 인코더로 문장 -> 임베딩
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        # [batch_size, seq_len, hidden_dim] 중 CLS pooling 또는 mean pooling
        # 여기서는 간단히 mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)

        # 2) Autoencoder로 차원 축소
        reduced_embeddings = self.autoencoder(embeddings)  # [batch_size, latent_dim]

        # 3) 양자 레이어(VQE)에서 나오는 양자 Loss
        quantum_loss = self.quantum_layer(reduced_embeddings)

        # 4) (예시) 최종 output으로 단어 분포 예측
        # 실제로는 번역 task에서는 seq2seq 구조가 필요하지만, 
        # 여기서는 단순 클래스 예측 예시만
        logits = self.output_layer(reduced_embeddings)  # [batch_size, vocab_size]

        return logits, quantum_loss

###################################
# 4. 학습 루프
###################################
def train_loop(model, dataloader, optimizer, criterion, device='cpu', lambda_q=0.1):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # (여기서는 label을 단순히 가장 첫 토큰을 예측하거나 etc. 하는 예시)
        # 실제 번역을 하려면 seq2seq loss를 구성해야 함
        # 일단 CrossEntropyLoss 위해서 label 차원을 맞춰줌
        if labels.ndim > 1:
            # 토큰 레벨의 CrossEntropy를 위해 (batch_size * seq_len) 모양으로 변환할 수도 있음
            # 여기서는 단순히 argmax(dim=1) 처리(아주 단순한 예시)
            labels = labels[:, 0]  # 첫 번째 토큰만 사용

        logits, quantum_loss = model(input_ids, attention_mask)
        # logits: [batch_size, vocab_size]
        # labels: [batch_size]
        ce_loss = criterion(logits, labels)

        loss = ce_loss + lambda_q * quantum_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} - Loss: {loss.item():.4f} (CE={ce_loss.item():.4f}, Q={quantum_loss.item():.4f})")

    return total_loss / len(dataloader)

def eval_loop(model, dataloader, criterion, device='cpu', lambda_q=0.1):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if labels.ndim > 1:
                labels = labels[:, 0]

            logits, quantum_loss = model(input_ids, attention_mask)
            ce_loss = criterion(logits, labels)

            loss = ce_loss + lambda_q * quantum_loss
            total_loss += loss.item()

    return total_loss / len(dataloader)


###################################
# 5. 메인 실행
###################################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -- Dataset, Dataloader 준비 (본 예시에서는 위에서 불러온다고 가정)
    # train_dataset, val_dataset, tokenizer 등이 준비되어 있다고 가정
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # -- BERT 설정
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.num_attention_heads = 2
    config.hidden_size = 128  # 임베딩 차원 조금 줄이기
    config.intermediate_size = 256
    encoder_model = BertModel(config).to(device)

    # -- Autoencoder
    latent_dim = 10
    autoencoder = Autoencoder(input_dim=config.hidden_size, latent_dim=latent_dim).to(device)

    # -- Quantum Layer
    quantum_circuit_module = QuantumCircuitWithParams(num_qubits=latent_dim)
    quantum_layer = VQEQuantumLayer(circuit_module=quantum_circuit_module)

    # -- 최종 모델
    vocab_size = tokenizer.vocab_size
    hybrid_model = HybridModel(
        encoder=encoder_model,
        autoencoder=autoencoder,
        quantum_layer=quantum_layer,
        vocab_size=vocab_size
    ).to(device)

    # -- Optimizer, Loss
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # -- 학습
    epochs = 3
    lambda_q = 0.1  # Quantum Loss 가중치
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_loop(hybrid_model, train_loader, optimizer, criterion, device=device, lambda_q=lambda_q)
        val_loss = eval_loop(hybrid_model, val_loader, criterion, device=device, lambda_q=lambda_q)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
