import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net_basic import Model
from transformers import AdamW, BertTokenizer
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 30000 
LEARNING_RATE = 5e-4

tokenizer = BertTokenizer.from_pretrained("../model/bert-base-uncased/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")

train_dataset = MyDataset("ag_news_train")
val_dataset = MyDataset("ag_news_test")

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

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True,
                          drop_last=True, # drop the last batch if it is not complete
                          collate_fn=custom_collate_fn)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=32,
                          shuffle=True,
                          drop_last=True, # drop the last batch if it is not complete
                          collate_fn=custom_collate_fn)

if __name__ == '__main__':
    print(f"Device: {str(DEVICE)}")
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)  # AdamW optimizer, recommended learning rates are 5e-5, 3e-5, or 2e-5 accordingly
    loss_func = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss

    model.train()  # set model to training mode
    # Training is happening here!!!!!!
    for epoch in range(EPOCH):
        sum_val_acc = 0
        sum_val_loss = 0
        best_avg_val_acc = 0
        sum_train_acc = 0
        sum_train_loss = 0
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
                DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids) # forward pass

            loss = loss_func(out, labels)

            optimizer.zero_grad()  # zero the gradients
            loss.backward()  # backward propagation
            optimizer.step()  # update the weights

            out = out.argmax(dim=1)
            acc_train = (out == labels).sum().item() / len(labels)
            sum_train_loss = sum_train_loss + loss
            sum_train_acc = sum_train_acc + acc_train
            
            # logging the training process
            if i % 5 == 0: # print every 5 batches
                print(f"training==>epoch:{epoch}, batch:{i}, loss:{loss.item()}, accuracy:{acc_train}")
        
        avg_train_loss = sum_train_loss / len(train_loader)
        avg_train_acc = sum_train_acc / len(train_loader)
        print(f"training==>epoch:{epoch},avg_train_loss:{avg_train_loss}, avg_train_acc:{avg_train_acc}")
                
        # Validation is happening here!!!!!!
        model.eval()
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
                DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids)

            loss = loss_func(out, labels)
            
            out = out.argmax(dim=1)
            acc_val = (out == labels).sum().item() / len(labels)
            sum_val_loss = sum_val_loss + loss
            sum_val_acc = sum_val_acc + acc_val
        
        avg_val_loss = sum_val_loss / len(val_loader)
        avg_val_acc = sum_val_acc / len(val_loader)
        print(f"validation==>epoch:{epoch},avg_val_loss:{avg_val_loss}, avg_val_acc:{avg_val_acc}")
        
        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", avg_val_acc, epoch)

        # if avg_val_acc > best_avg_val_acc:
        #     best_avg_val_acc = avg_val_acc
        torch.save(model.state_dict(), f"../trained_models/params/bert-basic-{epoch}.pth") # save the model at each epoch
        print(f"Model saved at epoch {epoch}!!!")
    
    writer.close()
