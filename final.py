from transformers import BertConfig, AutoTokenizer, BertModel
import torch
from torch.utils.data import DataLoader
from torch import nn
import os
from Data import train_dataset, val_dataset
from finalQunatum import decode_with_quantum_circuit, VQEQuantumLayer, QuantumCircuitWithParams

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# GPU 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 배치 크기 설정
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

# NLP 모델 초기화
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=64)
config = BertConfig.from_pretrained("bert-base-uncased")
config.num_attention_heads = 1
config.hidden_size = 64
config.intermediate_size = 256
model = BertModel(config).to(device)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# VQE Quantum Layer
quantum_layer = VQEQuantumLayer(circuit=None, optimizer=None, estimator=None)  # Initialize appropriately


# 출력층 추가
class HybridModel(nn.Module):
    def __init__(self, encoder, autoencoder, quantum_layer, num_classes):
        super(HybridModel, self).__init__()
        self.encoder = encoder
        self.autoencoder = autoencoder
        self.quantum_layer = quantum_layer
        self.output_layer = nn.Linear(autoencoder.encoder[-1].out_features, num_classes)  # 최종 출력층

    def forward(self, input_ids, attention_mask):
        embeddings = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
        reduced_embeddings = self.autoencoder(embeddings)
        logits = self.output_layer(reduced_embeddings)  # 클래스 출력으로 변환
        quantum_loss = self.quantum_layer(reduced_embeddings)
        return logits, quantum_loss


# 모델 초기화
latent_dim = 10
quantum_circuit = QuantumCircuitWithParams(latent_dim)  # 학습 가능한 양자 회로 생성
quantum_layer = VQEQuantumLayer(circuit=quantum_circuit, optimizer=None, estimator=None)  # VQE Quantum Layer 생성
autoencoder = Autoencoder(input_dim=64, latent_dim=latent_dim).to(device)
num_classes = tokenizer.vocab_size
hybrid_model = HybridModel(
    encoder=model, 
    autoencoder=autoencoder, 
    quantum_layer=quantum_layer, 
    num_classes=num_classes
).to(device)
# 옵티마이저 및 손실 함수
optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 학습 루프
def train_model():
    hybrid_model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        logits, quantum_loss = hybrid_model(input_ids, attention_mask)

        classical_loss = criterion(logits, labels)
        total_loss = classical_loss + 0.1 * quantum_loss  # VQE의 영향 감소


        total_loss.backward()
        optimizer.step()

        print(f"Loss: {total_loss.item()}")



# 평가 루프
def evaluate_model():
    hybrid_model.eval()
    cosine_similarities = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits, _ = hybrid_model(input_ids, attention_mask)
            reduced_embeddings = logits.detach().cpu().numpy()

            for embedding in reduced_embeddings:
                trainable_params = hybrid_model.quantum_layer.circuit.trainable_params.detach().cpu().numpy()
                quantum_decoded = decode_with_quantum_circuit(embedding, trainable_params)
                quantum_tensor = torch.tensor(quantum_decoded, dtype=torch.float32).unsqueeze(0)

                similarity = torch.cosine_similarity(
                    quantum_tensor,
                    torch.tensor(reduced_embeddings, dtype=torch.float32),
                    dim=1
                )
                cosine_similarities.append(similarity.mean().item())

    average_similarity = sum(cosine_similarities) / len(cosine_similarities)
    print(f"Average Cosine Similarity: {average_similarity}")


# 실행
if __name__ == "__main__":
    train_model()
    evaluate_model()
