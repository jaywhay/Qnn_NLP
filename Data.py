'''import os
import torch
import pandas as pd
import spacy
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# -------------------------
# 1. 파일 경로 설정 & 데이터 로드
# -------------------------
base_path = os.path.dirname(os.path.abspath(__file__))  
csv_file = os.path.join(base_path, "Data", "WMT21_en_de.csv")
df = pd.read_csv(csv_file)

# 필요한 열 추출
df = df[['source', 'eTranslation']]

def clean_sentence(sentence):
    return sentence.strip().lower()

df['source'] = df['source'].apply(clean_sentence)
df['eTranslation'] = df['eTranslation'].apply(clean_sentence)

# -------------------------
# 2. 의존 구문 분석 + 핵심 관계 추출
# -------------------------
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])  # NER, 분류 비활성화 → 속도 향상

def dependency_graph(sentence, min_words=12):
    """ 핵심 관계를 확장하여 정보 손실 방지 """
    doc = nlp(sentence)
    filtered_words = []

    for token in doc:
        if token.dep_ in ["nsubj", "prep", "amod", "dobj", "ROOT", "advmod", "pobj", "attr", "ccomp", "acl", "xcomp", "nmod", "conj", "appos"]:
            filtered_words.append(token.text)

    return " ".join(filtered_words) if len(filtered_words) >= min_words else sentence


if __name__ == '__main__':  # Windows 환경에서 필수!
    # -------------------------
    # 1. 파일 경로 설정 & 데이터 로드
    # -------------------------
    base_path = os.path.dirname(os.path.abspath(__file__))  
    csv_file = os.path.join(base_path, "Data", "WMT21_en_de.csv")
    df = pd.read_csv(csv_file)

    # 필요한 열 추출
    df = df[['source', 'eTranslation']]

    def clean_sentence(sentence):
        return sentence.strip().lower()

    df['source'] = df['source'].apply(lambda x: dependency_graph(x, min_words=8))
    df['eTranslation'] = df['eTranslation'].apply(lambda x: dependency_graph(x, min_words=8))

    # -------------------------
    # 3. 문장 길이 필터링
    # -------------------------
    min_len, max_len = 5, 30  # 문장 최소 길이 조정하여 너무 짧은 문장 제거
    df = df[df['source'].str.len().between(min_len, max_len)]
    df = df[df['eTranslation'].str.len().between(min_len, max_len)]

    # 빈 값이 있는 행 제거 (에러 방지)
    df.dropna(subset=['source', 'eTranslation'], inplace=True)

    # -------------------------
    # 4. 토크나이저 설정 및 Dataset 생성
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    class TranslationDataset(Dataset):
        def __init__(self, sources, targets, tokenizer, max_length=64):
            self.sources = sources
            self.targets = targets
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.sources)

        def __getitem__(self, idx):
            source_text = self.sources[idx]
            target_text = self.targets[idx]

            encoded = self.tokenizer(
                [source_text, target_text],  
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            return {
                'input_ids': encoded['input_ids'][0],
                'attention_mask': encoded['attention_mask'][0],
                'labels': encoded['input_ids'][1],
                'label_mask': encoded['attention_mask'][1]
            }

    # -------------------------
    # 5. Train / Validation 분리 및 DataLoader 추가
    # -------------------------
    train_src, val_src, train_tgt, val_tgt = train_test_split(
        df['source'].tolist(),
        df['eTranslation'].tolist(),
        train_size=0.8,
        random_state=42
    )

    train_dataset = TranslationDataset(train_src, train_tgt, tokenizer, max_length=16)
    val_dataset = TranslationDataset(val_src, val_tgt, tokenizer, max_length=16)

    # DataLoader 추가 (최종 학습에 필요)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    sample_batch = next(iter(train_loader))

    print("Input IDs:", sample_batch['input_ids'][0])  # 첫 번째 샘플
    print("Attention Mask:", sample_batch['attention_mask'][0])
    print("Labels:", sample_batch['labels'][0])
'''

import os
import json
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

# -------------------------
# 1. 파일 경로 설정 & 데이터 로드
# -------------------------
base_path = os.path.dirname(os.path.abspath(__file__))  
csv_file = os.path.join(base_path, "Data", "WMT21_en_de.csv")
df = pd.read_csv(csv_file)

# 필요한 열 추출
df = df[['source', 'eTranslation']]

# -------------------------
# 2. 데이터 전처리: 의존 구문 분석 적용
# -------------------------
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])  # 속도 향상을 위해 불필요한 태스크 제거

def dependency_graph(sentence, min_words=8):
    """ 핵심 관계를 추출하는 의존 구문 분석 수행 """
    doc = nlp(sentence)
    filtered_words = []

    for token in doc:
        if token.dep_ in ["nsubj", "prep", "amod", "dobj", "ROOT", "advmod", "pobj", "attr", "ccomp", "acl", "xcomp", "nmod", "conj", "appos"]:
            filtered_words.append(token.text)

    return " ".join(filtered_words) if len(filtered_words) >= min_words else sentence

df['source_processed'] = df['source'].apply(lambda x: dependency_graph(x, min_words=8))
df['eTranslation_processed'] = df['eTranslation'].apply(lambda x: dependency_graph(x, min_words=8))

# -------------------------
# 3. 데이터 저장 (JSON 형식)
# -------------------------
output_file = os.path.join(base_path, "processed_data.json")

data_dict = {
    "source": df['source'].tolist(),
    "source_processed": df['source_processed'].tolist(),
    "eTranslation": df['eTranslation'].tolist(),
    "eTranslation_processed": df['eTranslation_processed'].tolist()
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_dict, f, indent=4, ensure_ascii=False)