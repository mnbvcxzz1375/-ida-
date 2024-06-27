import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取数据
data_path = 'merged_data.csv'
data = pd.read_csv(data_path)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 分割数据集为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# 加载预训练的 ERNIE 模型和分词器
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-3.0-mini-zh")
ernie_model = AutoModel.from_pretrained("nghuyong/ernie-3.0-mini-zh")

# 假设 train_data 和 test_data 是已经加载的数据框，包含列 '0' 和 'label'
train_texts = train_data['0'].tolist()
train_labels = train_data['label'].tolist()

test_texts = test_data['0'].tolist()
test_labels = test_data['label'].tolist()


# 处理编码和截断
def preprocess_data(texts, tokenizer, max_length=128):
    encodings = tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']


train_inputs, train_masks = preprocess_data(train_texts, tokenizer)
test_inputs, test_masks = preprocess_data(test_texts, tokenizer)

train_labels = torch.tensor(train_labels).unsqueeze(1).expand(-1, train_inputs.size(1))
test_labels = torch.tensor(test_labels).unsqueeze(1).expand(-1, test_inputs.size(1))

# CRF层的实现
from torchcrf import CRF


# 构建模型
class TextClassifier(nn.Module):
    def __init__(self, ernie_model, hidden_size, num_labels, dropout_rate=0.5):
        super(TextClassifier, self).__init__()
        self.ernie_model = ernie_model
        self.lstm = nn.LSTM(ernie_model.config.hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            outputs = self.ernie_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        emissions = self.fc(lstm_output)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.byte())
            return prediction


# 定义模型和优化器
hidden_size = 128
num_labels = 2  # 假设有两个标签
dropout_rate = 0.5

model = TextClassifier(ernie_model, hidden_size, num_labels, dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# 转换为 DataLoader
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 训练模型
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for inputs, masks, labels in train_loader:
        optimizer.zero_grad()
        loss = model(inputs, attention_mask=masks, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()  # 更新学习率

    # 计算平均损失
    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    test_inputs = test_inputs.to(torch.long)
    test_masks = test_masks.to(torch.long)
    test_labels = test_labels.to(torch.long)

    predictions = model(test_inputs, attention_mask=test_masks)
    correct = 0
    total = 0
    for prediction, label in zip(predictions, test_labels):
        correct += (torch.tensor(prediction) == label).sum().item()
        total += len(label)

    test_accuracy = correct / total

# 打印评估结果
print('Test Accuracy:', test_accuracy)
