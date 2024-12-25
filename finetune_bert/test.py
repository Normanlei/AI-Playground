import torch
from MyData import MyDataset
from torch.utils.data import DataLoader, Subset
# from net_basic import Model
from net_config import Model
from transformers import BertTokenizer
from sklearn.metrics import classification_report, accuracy_score
from utils import BERT_MODEL_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'we are at device {str(DEVICE)}')
DATASET_NAME = "imdb" # imdb or ag_news
epoch = 0 # epoch number of the saved model
MODEL_PATH = f"../trained_models/bert-{DATASET_NAME}-{epoch}.pth"  # Path to the saved model parameters
BATCH_SIZE = 32

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

# Load test dataset
test_dataset = MyDataset(f"{DATASET_NAME}_test")
test_dataset = Subset(test_dataset, range(5000)) # for testing mem saving purposes
print(f"test_dataset size: {len(test_dataset)}")

def custom_collate_fn(data):
    texts = [i[0] for i in data] 
    labels = [i[1] for i in data]
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                       truncation=True,
                                       padding="max_length",
                                       return_tensors="pt",
                                       return_length=True) 
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"] 
    token_type_ids = data["token_type_ids"]  
    collated_labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, collated_labels

# DataLoader for test data
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         collate_fn=custom_collate_fn)
print(f"test_loader size: {len(test_loader)}")

# Load model and weights
model = Model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Evaluate on test data
all_preds = []
all_labels = []

with torch.no_grad():
    for input_ids, attention_mask, token_type_ids, labels in test_loader:
        input_ids, attention_mask, token_type_ids, labels = (
            input_ids.to(DEVICE),
            attention_mask.to(DEVICE),
            token_type_ids.to(DEVICE),
            labels.to(DEVICE)
        )
        outputs = model(input_ids, attention_mask, token_type_ids)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
