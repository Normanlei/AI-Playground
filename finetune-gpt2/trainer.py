from transformers import AdamW
from transformers.optimization import get_scheduler # dynamic learning rate scheduler
import torch
from MyData import MyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

dataset = MyDataset()
print('dataset length:', len(dataset))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'we are at device {str(DEVICE)}')
EPOCH = 3 
LEARNING_RATE = 2e-5
MODEL_PATH = ''
BATCH_SIZE = 

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# print(model)

def collate_fn(data):
    data = tokenizer.batch_encode_plus(data, padding=True, truncation=True, max_length=512, return_tensors='pt')

    data['labels'] = data['input_ids'].clone()

    return data

train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
)
print('loader length:', len(train_loader))    

if __name__ == '__main__':
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)  # AdamW optimizer, recommended learning rates are 5e-5, 3e-5, or 2e-5 accordingly
    scheduler = get_scheduler(name='linear', num_warmup_steps=0, num_training_steps=len(train_loader), optimizer=optimizer)

    model.train()
    for epoch in range(EPOCH):
        for i, data in enumerate(loader):
        for k in data.keys():
            data[k] = data[k].to(DEVICE)
        out = model(**data) # forward pass
        
        loss = out['loss'] # gpt2 integrated loss

        loss.backward() # backward propagation
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping to prevent exploding gradients

        optimizer.step() # update the weights
        scheduler.step() # update the learning rate

        optimizer.zero_grad() # zero the gradients
        model.zero_grad()

        # logging the training process
        if i % 50 == 0:
            # token-level prediction accuracy
            labels = data['labels'][:, 1:] # omit the first token <s>
            out = out['logits'].argmax(dim=2)[:, :-1] # omit the last token </s>

            select = labels != 0 # filter out the padding tokens
            labels = labels[select]
            out = out[select]
            del select
            
            accuracy = (labels == out).sum().item() / labels.numel()

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            
            perplexity = torch.exp(loss) # perplexity is another metric to evaluate the model, the lower the better

            print(f"training==>epoch:{epoch}, batch:{i}, learn_rate:{lr}, perplexity:{perplexity.item()}, token-level accuracy:{accuracy}") 

        torch.save(model.state_dict(), f"../trained_models/gpt2-chinese-{epoch}.pth") # save the model at each epoch
        print(f"Model saved at epoch {epoch}!!!")
