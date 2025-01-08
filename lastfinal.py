import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, AutoTokenizer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

# Dataset 및 전처리
class TranslationDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_length=64):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src_text = self.sources[idx]
        tgt_text = self.targets[idx]

        src_enc = self.tokenizer(
            src_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'src_input_ids': src_enc['input_ids'].squeeze(0),
            'src_attention_mask': src_enc['attention_mask'].squeeze(0),
            'tgt_input_ids': tgt_enc['input_ids'].squeeze(0),
            'tgt_attention_mask': tgt_enc['attention_mask'].squeeze(0),
        }

def clean_text(s: str) -> str:
    return s.strip().lower()

def load_and_prepare_dataset(csv_file="Data/WMT14_en_de.csv", max_len=64):
    df = pd.read_csv(csv_file)
    df = df[['source', 'eTranslation']]
    df['source'] = df['source'].apply(clean_text)
    df['eTranslation'] = df['eTranslation'].apply(clean_text)

    min_len, max_len_th = 5, 50
    df = df[df['source'].str.split().apply(len).between(min_len, max_len_th)]
    df = df[df['eTranslation'].str.split().apply(len).between(min_len, max_len_th)]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_src, val_src, train_tgt, val_tgt = train_test_split(
        df['source'].tolist(),
        df['eTranslation'].tolist(),
        train_size=0.8,
        random_state=42
    )

    train_dataset = TranslationDataset(train_src, train_tgt, tokenizer, max_length=max_len)
    val_dataset = TranslationDataset(val_src, val_tgt, tokenizer, max_length=max_len)

    return train_dataset, val_dataset, tokenizer

# Autoencoder (Encoder만, latent_dim=10)
class Autoencoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# 스피어만 상관계수 기반 Sparse Hamiltonian
def build_spearman_correlation_matrix(embedding_samples: np.ndarray, threshold=0.7):
    corr, _ = spearmanr(embedding_samples, axis=0)
    high_corr_mask = (np.abs(corr) >= threshold)
    return corr, high_corr_mask

def generate_sparse_hamiltonian(z_vec: torch.Tensor,
                                corr_matrix: np.ndarray,
                                high_corr_mask: np.ndarray):
    """
    z_vec: shape (latent_dim,)
    """
    z_np = z_vec.detach().cpu().numpy()
    n = len(z_np)

    paulis = []
    # (A) 단일 Z 항
    for i in range(n):
        label = ["I"]*n
        label[i] = "Z"
        paulis.append(("".join(label), float(z_np[i])))

    # (B) ZZ 상관 항
    for i in range(n):
        for j in range(i+1, n):
            if high_corr_mask[i, j]:
                label = ["I"]*n
                label[i] = "Z"
                label[j] = "Z"
                coeff = corr_matrix[i, j] * z_np[i] * z_np[j]
                paulis.append(("".join(label), float(coeff)))

    if len(paulis) == 0:
        return None
    return SparsePauliOp.from_list(paulis)

class QuantumCircuitWithParams(nn.Module):
    def __init__(self, num_qubits=10):
        super(QuantumCircuitWithParams, self).__init__()
        self.num_qubits = num_qubits
        self.trainable_params = nn.Parameter(torch.rand(num_qubits))

    def forward(self, data_vec: torch.Tensor) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        for i in range(self.num_qubits):
            theta = float(self.trainable_params[i].item())
            val = float(data_vec[i].item())
            qc.ry(val * theta, i)
            qc.rz(val * theta / 2.0, i)

        # 인접 CNOT
        for i in range(self.num_qubits - 1):
            qc.cx(i, i+1)

        return qc

class VQEQuantumLayer(nn.Module):
    def __init__(self, circuit_module: QuantumCircuitWithParams, corr_matrix: np.ndarray, high_corr_mask: np.ndarray, estimator=None):
        super(VQEQuantumLayer, self).__init__()
        self.circuit_module = circuit_module
        self.corr_matrix = corr_matrix
        self.high_corr_mask = high_corr_mask
        self.estimator = estimator if estimator is not None else Estimator()
        self.simulator = AerSimulator(method='statevector')

    def forward(self, batch_z: torch.Tensor):
        energies = []
        for i in range(batch_z.size(0)):
            z_vec = batch_z[i]
            H = generate_sparse_hamiltonian(z_vec, self.corr_matrix, self.high_corr_mask)
            if H is None:
                energies.append(0.0)
                continue

            qc = self.circuit_module(z_vec)
            qc_ = transpile(qc, self.simulator, optimization_level=1)

            try:
                job = self.estimator.run([qc_], [H])
                result = job.result()
                energy = result.values[0]
            except Exception as e:
                print(f"[Estimator Error] idx={i}, {e}")
                energy = 0.0

            energies.append(energy)

        if len(energies) == 0:
            return torch.tensor(0.0)
        return torch.tensor(energies, dtype=torch.float32).mean()

class HybridEncoder(nn.Module):
    def __init__(self, bert_encoder: BertModel, autoencoder: Autoencoder, quantum_layer: VQEQuantumLayer, output_dim=128):
        super(HybridEncoder, self).__init__()
        self.bert_encoder = bert_encoder
        self.autoencoder = autoencoder
        self.quantum_layer = quantum_layer

        # 최종 인코더 출력 차원을 Seq2Seq 디코더와 맞추기 위해 10차원 -> 128차원 등으로 매핑
        self.final_proj = nn.Linear(10, output_dim)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert_encoder(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch, 128]

        z = self.autoencoder(embeddings)                    # [batch, 10]

        vqe_loss = self.quantum_layer(z)

        encoder_out = self.final_proj(z)  # [batch, output_dim=128]

        return encoder_out, vqe_loss
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim=128, nhead=4, dim_feedforward=256):
        super().__init__()
        from torch.nn import TransformerDecoderLayer
        self.layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # PyTorch >=1.10
        )

    def forward(self, tgt, memory, tgt_mask=None,
                memory_key_padding_mask=None, tgt_key_padding_mask=None):
        return self.layer(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

class TranslationTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=2, nhead=4, dim_feedforward=256):
        super().__init__()
        from torch.nn import TransformerDecoder
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer.layer, num_layers=num_layers)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_tokens, memory, tgt_mask=None):
        # 임베딩
        tgt_emb = self.embedding(tgt_tokens)  # [batch, tgt_len, hidden_dim]
        # TransformerDecoder
        out = self.transformer_decoder(
            tgt_emb,  # [batch, tgt_len, hidden_dim]
            memory,   # [batch, src_len, hidden_dim]
            tgt_mask=tgt_mask
        )  # [batch, tgt_len, hidden_dim]
        # 로짓
        logits = self.fc_out(out)  # [batch, tgt_len, vocab_size]
        return logits

class QuantumSeq2Seq(nn.Module):
    def __init__(self, encoder: HybridEncoder, decoder: TranslationTransformerDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_mask=None):
        # (A) 인코더
        encoder_out, vqe_loss = self.encoder(src_input_ids, src_attention_mask)
        # encoder_out shape: [batch, enc_len=1 or ???, hidden_dim=128] 
        # 실제론 seq_len이 1이 아니라 BERT의 seq_len이 될 수도 있음.
        # 여기선 mean pooling이므로 (batch, 128) 형태.
        # TransformerDecoder는 (batch, enc_seq_len, hidden_dim) 이 필요..?
        # 일단 enc_seq_len=1로 간주 -> (batch, 1, 128)
        encoder_out = encoder_out.unsqueeze(1)

        # (B) 디코더
        logits = self.decoder(tgt_input_ids, encoder_out, tgt_mask=tgt_mask)

        return logits, vqe_loss

def train_seq2seq(model: QuantumSeq2Seq, dataloader, optimizer, vocab_size, device='cpu', lambda_q=0.1):
    model.train()
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0= [PAD] 가정

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        src_ids = batch['src_input_ids'].to(device)
        src_mask = batch['src_attention_mask'].to(device)
        tgt_ids = batch['tgt_input_ids'].to(device)   # [batch, seq_len]

        # Teacher forcing: tgt_ids[:, :-1] -> input, tgt_ids[:, 1:] -> label
        decoder_input = tgt_ids[:, :-1]
        decoder_label = tgt_ids[:, 1:].clone()

        # forward
        logits, vqe_loss = model(src_ids, src_mask, decoder_input)

        # logits: [batch, tgt_len-1, vocab_size]
        # decoder_label: [batch, tgt_len-1]
        logits_2d = logits.reshape(-1, vocab_size)
        labels_1d = decoder_label.reshape(-1)

        ce_loss = criterion(logits_2d, labels_1d)
        loss = ce_loss + lambda_q * vqe_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f"[Train step {step}] total_loss={loss.item():.4f} "f"(CE={ce_loss.item():.4f}, VQE={vqe_loss.item():.4f})")

    return total_loss / len(dataloader)

@torch.no_grad()
def eval_seq2seq(model: QuantumSeq2Seq, dataloader, vocab_size, device='cpu', lambda_q=0.1):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for step, batch in enumerate(dataloader):
        src_ids = batch['src_input_ids'].to(device)
        src_mask = batch['src_attention_mask'].to(device)
        tgt_ids = batch['tgt_input_ids'].to(device)

        decoder_input = tgt_ids[:, :-1]
        decoder_label = tgt_ids[:, 1:].clone()

        logits, vqe_loss = model(src_ids, src_mask, decoder_input)
        logits_2d = logits.reshape(-1, vocab_size)
        labels_1d = decoder_label.reshape(-1)

        ce_loss = criterion(logits_2d, labels_1d)
        loss = ce_loss + lambda_q * vqe_loss
        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def translate_greedy(model: QuantumSeq2Seq, tokenizer, src_text: str, max_len=40, device='cpu'):
    model.eval()
    # 소스 인코딩
    enc = tokenizer(src_text, return_tensors='pt')
    src_ids = enc['input_ids'].to(device)
    src_mask = enc['attention_mask'].to(device)

    # 인코더만 실행
    encoder_out, _ = model.encoder(src_ids, src_mask)
    # shape: [batch=1, 128] -> (1, 1, 128)
    encoder_out = encoder_out.unsqueeze(1)

    # 디코더: [BOS] 토큰부터 시작
    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id else 101
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id else 102
    current_tokens = [bos_id]

    for _ in range(max_len):
        decoder_input = torch.tensor([current_tokens], dtype=torch.long, device=device)
        # [1, seq_len]
        logits = model.decoder(decoder_input, encoder_out)  # [1, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :]  # 마지막 스텝
        next_token_id = int(next_token_logits.argmax(dim=-1).item())

        if next_token_id == eos_id:
            break
        current_tokens.append(next_token_id)

    # 토큰 시퀀스를 tokenizer.decode
    return tokenizer.decode(current_tokens[1:], skip_special_tokens=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 데이터셋
    train_dataset, val_dataset, tokenizer = load_and_prepare_dataset("Data/WMT14_en_de.csv", max_len=32)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # BERT 인코더
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.hidden_size = 128
    config.num_attention_heads = 2
    config.intermediate_size = 256
    bert_encoder = BertModel(config).to(device)

    # Autoencoder
    autoencoder = Autoencoder(input_dim=128, latent_dim=10).to(device)

    # Spearman Corr
    sample_z = []
    sampling_batches = 3
    for i, batch in enumerate(train_loader):
        if i >= sampling_batches:
            break
        with torch.no_grad():
            src_ids = batch['src_input_ids'].to(device)
            src_mask = batch['src_attention_mask'].to(device)
            out = bert_encoder(src_ids, attention_mask=src_mask).last_hidden_state.mean(dim=1)  # [B, 128]
            z_ = autoencoder(out)
            sample_z.append(z_.cpu().numpy())

    if len(sample_z) > 0:
        allz = np.concatenate(sample_z, axis=0)
        corr_matrix, high_corr_mask = build_spearman_correlation_matrix(allz, threshold=0.7)
    else:
        print("[Warning] Not enough data for correlation. Use identity.")
        corr_matrix = np.eye(10)
        high_corr_mask = (np.abs(corr_matrix) >= 0.7)

    # Quantum Circuit & VQE
    circuit_module = QuantumCircuitWithParams(num_qubits=10).to(device)
    quantum_layer = VQEQuantumLayer(circuit_module, corr_matrix, high_corr_mask).to(device)

    # HybridEncoder
    hybrid_encoder = HybridEncoder(
        bert_encoder=bert_encoder,
        autoencoder=autoencoder,
        quantum_layer=quantum_layer,
        output_dim=128
    ).to(device)

    # Transformer Decoder
    vocab_size = tokenizer.vocab_size
    decoder = TranslationTransformerDecoder(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=2,
        nhead=4,
        dim_feedforward=256
    ).to(device)

    # 전체 Seq2Seq 모델
    model = QuantumSeq2Seq(hybrid_encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lambda_q = 0.1
    epochs = 2

    print("\n=== Training Seq2Seq Model ===")
    for epoch in range(epochs):
        train_loss = train_seq2seq(model, train_loader, optimizer, vocab_size, device=device, lambda_q=lambda_q)
        val_loss = eval_seq2seq(model, val_loader, 
                                vocab_size, device=device, lambda_q=lambda_q)
        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    print("\n=== Testing Greedy Translation ===")
    test_sentences = [
        "This is a test sentence",
        "I like quantum computing"
    ]
    for sent in test_sentences:
        translation = translate_greedy(model, tokenizer, sent, max_len=20, device=device)
        print(f"Source: {sent}")
        print(f"Target(Pred): {translation}")
        print("")


if __name__ == "__main__":
    main()