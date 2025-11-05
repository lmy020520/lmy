#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗器械问句 6 分类 微调脚本
单机单卡 / 多卡都可跑
"""
import os, json, random, warnings, math
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
set_seed(42)
warnings.filterwarnings("ignore")
import pandas as pd

df = pd.read_csv("/home/lmy/study/lmy/二级医疗器械/train.csv", quotechar='"', escapechar='\\')
print(df.columns)

# ---------- 1. 参数区（只改这里就行） ----------
MODEL_NAME = "/home/lmy/study/lmy/Model/model4_Chinese-RoBERTa-wwm-ext"  # 你本地权重
TRAIN_FILE = "/home/lmy/study/lmy/二级医疗器械/train.csv"
DEV_FILE   = "/home/lmy/study/lmy/二级医疗器械/dev.csv"
OUTPUT_DIR = "nmpa_roberta_cls"
NUM_LABELS = 6
MAX_LEN    = 128
BATCH_SIZE = 16
LR         = 2e-5
EPOCHS     = 5
WARMUP     = 0.1
# ----------------------------------------------

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

class QCDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, encoding='utf-8-sig', on_bad_lines='skip')
        # 去掉可能出现的空列
        df = df.dropna(axis=1, how='all')
        self.texts  = df['question'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding=False,
            truncation=True,
        )
        enc['labels'] = self.labels[idx]
        return enc

train_ds = QCDataset(TRAIN_FILE)
dev_ds   = QCDataset(DEV_FILE)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_ratio=WARMUP,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    # 旧版兼容写法 ↓
    eval_strategy="steps",   # 如果提示仍不识别，就改成 eval_strategy="steps"
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n===== 最终在 dev 上的指标 =====")
metrics = trainer.evaluate()
print(json.dumps(metrics, indent=2, ensure_ascii=False))