from transformers import BertModel 
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = BertModel.from_pretrained("../model/bert-base-uncased/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594").to(DEVICE)
print(pretrained)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 4)  # add a fully connected layer to classify the extracted features ag_news dataset has 4 classes

    def forward(self,input_ids,attention_mask,token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:,0])
        out = out.softmax(dim=1)
        return out