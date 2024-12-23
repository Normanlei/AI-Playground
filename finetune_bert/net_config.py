from transformers import BertModel, BertConfig 
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configuration = BertConfig.from_pretrained("../model/bert-base-uncased/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
configuration.max_position_embeddings = 1500  # increase the maximum position embedding to 1500 to accommodate longer sequences in imdb dataset
print(configuration)

pretrained = BertModel(configuration).to(DEVICE)
print(pretrained.embeddings.position_embeddings)  
print(pretrained)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2) # add a fully connected layer to classify the extracted features imdb dataset has 2 classes
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # train embeddings layer since we are using a new configuration with a new position embedding
        embeddings_output = pretrained.embeddings(input_ids=input_ids)
        attention_mask = attention_mask.to(torch.float)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # add dimensions to attention mask to match the shape of embeddings_output
        attention_mask = attention_mask.to(embeddings_output.dtype)  # convert attention mask to the same data type as embeddings_output
        with torch.no_grad():
            encoder_output = pretrained.encoder(embeddings_output, attention_mask=attention_mask) # token_type_ids is optional??
        out = self.fc(encoder_output.last_hidden_state[:, 0]) 
        return out