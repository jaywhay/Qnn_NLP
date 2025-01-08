from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Depth, Optimize1qGates, CommutativeCancellation
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator
from joblib import Parallel, delayed
import torch
import numpy as np

simulator = AerSimulator()

# 데이터 기반 해밀토니안 생성
def generate_hamiltonian_from_data(data_vectors):
    if isinstance(data_vectors, torch.Tensor) and data_vectors.ndim == 1:
        data_vectors = torch.clamp(data_vectors, min=1e-6)  # 너무 작은 값을 제거

    hamiltonians = []
    for idx, data_vector in enumerate(data_vectors):
        if len(data_vector) == 0:  # 비어 있는 벡터 처리
            print(f"Empty data vector at index {idx}")
            hamiltonians.append(None)
            continue

        try:
            num_qubits = len(data_vector)
            terms = [
                (f"{'Z' * num_qubits}", float(coeff.item()) if isinstance(coeff, torch.Tensor) else float(coeff))
                for coeff in data_vector
            ]
            hamiltonians.append(SparsePauliOp.from_list(terms))
        except Exception as e:
            print(f"Error generating Hamiltonian for vector {idx}: {e}")
            hamiltonians.append(None)

    return hamiltonians


# T-depth 최적화
def optimize_t_depth(circuit):
    transpiled_circuit = transpile(circuit, backend=simulator, optimization_level=3)
    pass_manager = PassManager([
        Optimize1qGates(),
        CommutativeCancellation(),
        Depth()
    ])
    optimized_circuit = pass_manager.run(transpiled_circuit)
    return optimized_circuit

#2024 12 23
def create_data_encoded_circuit(data_vector, num_qubits):

    circuit = QuantumCircuit(num_qubits)
    parameters = data_vector  # 데이터 벡터를 회로의 파라미터로 사용

    # 각 큐비트에 데이터 기반 RY, RZ 적용
    for i in range(num_qubits):
        circuit.ry(parameters[i], i)
        circuit.rz(parameters[i] / 2, i)
    
    # T 게이트와 CNOT 게이트를 추가하여 엔탱글먼트 생성
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    circuit.t(num_qubits - 1)  # 마지막 큐비트에 T 게이트 적용

    return circuit

def decode_with_quantum_circuit(reduced_embedding, trainable_params):
    num_qubits = min(len(reduced_embedding), len(trainable_params))
    circuit = QuantumCircuit(num_qubits)

    # 데이터 기반 초기화 및 회전 게이트 적용
    for i, val in enumerate(reduced_embedding[:num_qubits]):
        circuit.rx(float(val * trainable_params[i]), i)
        circuit.rz(float(val / 2 * trainable_params[i]), i)

    # 데이터 간 상호작용: CNOT 게이트
    for i in range(num_qubits - 1):
        if reduced_embedding[i] > reduced_embedding[i + 1]:
            circuit.cx(i, i + 1)
        else:
            circuit.cx(i + 1, i)

    # AerSimulator 실행
    simulator = AerSimulator(method="statevector")
    transpiled_circuit = transpile(circuit, simulator)

    try:
        result = simulator.run(transpiled_circuit).result()
        print("Simulation result data:", result.data())
    except Exception as e:
        print("Simulation error:", e)
        raise ValueError("Simulation failed.")

    # 상태 벡터 확인 및 반환
    if "statevector" not in result.data():
        raise ValueError("Statevector not found in simulation results.")

    statevector = result.get_statevector()
    return np.real(statevector)



class QuantumCircuitWithParams(torch.nn.Module):
    def __init__(self, num_qubits):
        super(QuantumCircuitWithParams, self).__init__()
        self.num_qubits = num_qubits
        self.trainable_params = torch.nn.Parameter(torch.rand(num_qubits))  # 학습 가능한 파라미터

    @property
    def num_parameters(self):
        return self.num_qubits  # VQE에서 참조할 파라미터 개수 반환

    def forward(self, reduced_embedding):
        circuit = QuantumCircuit(self.num_qubits)

        # 데이터 기반 게이트 적용
        for i, val in enumerate(reduced_embedding):
            circuit.rx(float(val * self.trainable_params[i].item()), i)
            circuit.ry(float(val / 2 * self.trainable_params[i].item()), i)

        # CNOT로 상호작용 반영
        for i in range(self.num_qubits - 1):
            if reduced_embedding[i] > reduced_embedding[i + 1]:
                circuit.cx(i, i + 1)
            else:
                circuit.cx(i + 1, i)

        # 시뮬레이터 사용
        simulator = AerSimulator(method="statevector")
        transpiled_circuit = transpile(circuit, simulator)
        result = simulator.run(transpiled_circuit).result()

        if "statevector" not in result.data():
            raise ValueError("Statevector not found in simulation results.")

        statevector = result.get_statevector()
        return torch.tensor(np.real(statevector), dtype=torch.float32)


class VQEQuantumLayer(torch.nn.Module):
    def __init__(self, circuit, optimizer=None, estimator=None):
        super(VQEQuantumLayer, self).__init__()
        self.circuit = circuit
        self.optimizer = optimizer or COBYLA()  # 기본 최적화기로 COBYLA 사용
        self.estimator = estimator or Estimator()  # 기본적으로 Primitives 기반 Estimator 사용

    def forward(self, input_data):
        hamiltonians = generate_hamiltonian_from_data(input_data)
        quantum_losses = []

        for idx, hamiltonian in enumerate(hamiltonians):
            if hamiltonian is None:
                continue

            try:
                generated_circuit = self.circuit.forward(input_data[idx])
                if not isinstance(generated_circuit, QuantumCircuit):
                    raise ValueError(f"Generated circuit is not a QuantumCircuit: {generated_circuit}")

                # Ensure the circuit is wrapped in a list
                result = self.estimator.run([generated_circuit], hamiltonian).result()
                quantum_losses.append(result.values[0])
            except Exception as e:
                print(f"VQE execution error at index {idx}: {e}")
                quantum_losses.append(None)

        quantum_losses = [loss for loss in quantum_losses if loss is not None]
        return torch.tensor(quantum_losses, dtype=torch.float32).mean() if quantum_losses else torch.tensor(0.0)

# 간단한 회로 생성
circuit = QuantumCircuit(1)
circuit.h(0)  # Hadamard 게이트 적용

# AerSimulator 설정
simulator = AerSimulator(method="statevector")
transpiled_circuit = transpile(circuit, simulator)
result = simulator.run(transpiled_circuit).result()

if "statevector" not in result.data():
    print("Error: Statevector not found in results.")
    raise ValueError("Statevector not found in simulation results.")
