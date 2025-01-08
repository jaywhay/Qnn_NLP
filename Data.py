import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# 파일 경로
csv_file = "Data\WMT21_en_de.csv"

# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 필요한 열 추출 (source, eTranslation)
df = df[['source', 'eTranslation']]

# 데이터 정제 함수
def clean_sentence(sentence):
    sentence = sentence.strip().lower()  # 소문자로 변환 및 공백 제거
    return sentence

# 데이터 정제 적용
df['source'] = df['source'].apply(clean_sentence)
df['eTranslation'] = df['eTranslation'].apply(clean_sentence)

# 문장 길이 필터링
min_len, max_len = 5, 50  # 최소 및 최대 길이 설정
df = df[df['source'].str.split().apply(len).between(min_len, max_len)]
df = df[df['eTranslation'].str.split().apply(len).between(min_len, max_len)]

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 토큰화
source_tokenized = tokenizer(df['source'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
target_tokenized = tokenizer(df['eTranslation'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")

# 데이터셋 클래스 정의 (수정 2번 째)
class TranslationDataset(Dataset):
    def __init__(self, source, target, source_mask, target_mask):
        self.source = source['input_ids']
        self.target = target['input_ids']
        self.source_mask = source_mask  # 추가
        self.target_mask = target_mask  # 추가

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return {
            'input_ids': self.source[idx],
            'labels': self.target[idx],
            'attention_mask': self.source_mask[idx],  # 추가
            'label_mask': self.target_mask[idx]  # 추가
        }

# 데이터셋 분리 후 리스트로 변환
train_size = 0.8
train_src, val_src, train_tgt, val_tgt = train_test_split(
    df['source'].tolist(), df['eTranslation'].tolist(), train_size=train_size, random_state=42
)

train_src_tokenized = tokenizer(train_src, padding=True, truncation=True, return_tensors="pt")
train_tgt_tokenized = tokenizer(train_tgt, padding=True, truncation=True, return_tensors="pt")
val_src_tokenized = tokenizer(val_src, padding=True, truncation=True, return_tensors="pt")
val_tgt_tokenized = tokenizer(val_tgt, padding=True, truncation=True, return_tensors="pt")

# DataLoader 생성에 필요한 TranslationDataset
train_dataset = TranslationDataset(
    train_src_tokenized, train_tgt_tokenized,
    train_src_tokenized['attention_mask'], train_tgt_tokenized['attention_mask']
)
val_dataset = TranslationDataset(
    val_src_tokenized, val_tgt_tokenized,
    val_src_tokenized['attention_mask'], val_tgt_tokenized['attention_mask']
)

