import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, BertConfig, BertModel
import torch.nn as nn

# -------------------------
# 1. CSV 로드 및 간단 전처리
# -------------------------
csv_file = "Data/WMT21_en_de.csv"
df = pd.read_csv(csv_file)

# 필요한 열 추출
df = df[['source', 'eTranslation']]

def clean_sentence(sentence):
    # 소문자로 변환, 앞뒤 공백 제거
    return sentence.strip().lower()

df['source'] = df['source'].apply(clean_sentence)
df['eTranslation'] = df['eTranslation'].apply(clean_sentence)

# 문장 길이 필터링
min_len, max_len = 5, 50
df = df[df['source'].str.split().apply(len).between(min_len, max_len)]
df = df[df['eTranslation'].str.split().apply(len).between(min_len, max_len)]

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# -------------------------
# 2. Dataset 정의
# -------------------------
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

        source_encoded = self.tokenizer(
            source_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        target_encoded = self.tokenizer(
            target_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )

        return {
            'input_ids': source_encoded['input_ids'].squeeze(0),
            'attention_mask': source_encoded['attention_mask'].squeeze(0),
            'labels': target_encoded['input_ids'].squeeze(0),
            'label_mask': target_encoded['attention_mask'].squeeze(0)
        }

# -------------------------
# 3. Train / Val 분리
# -------------------------
train_src, val_src, train_tgt, val_tgt = train_test_split(
    df['source'].tolist(),
    df['eTranslation'].tolist(),
    train_size=0.8,
    random_state=42
)

train_dataset = TranslationDataset(train_src, train_tgt, tokenizer, max_length=64)
val_dataset = TranslationDataset(val_src, val_tgt, tokenizer, max_length=64)
