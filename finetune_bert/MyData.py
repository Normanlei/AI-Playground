from torch.utils.data import Dataset
from datasets import load_dataset

class MyDataset(Dataset):
    def __init__(self,dataset_name):
        self.dataset = load_dataset(path="csv",data_files=f"../dataset/{dataset_name}.csv",split="train")
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]

        return text,label

if __name__ == '__main__':
    dataset = MyDataset("imdb_train") # Dataset name in ../dataset folder
    for data in dataset:
        print(data)