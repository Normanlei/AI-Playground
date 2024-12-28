from transformers import AutoModelForCausalLM,AutoTokenizer,TextGenerationPipeline
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'we are at device {str(DEVICE)}')
MODEL_PARAM_PATH = f"../trained_models/gpt2-chinese-4.pth"  # Path to the saved model parameters
MODEL_PATH = '../model/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3'


model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Load model and weights
model = model.to(DEVICE)
model.load_state_dict(torch.load(MODEL_PARAM_PATH, map_location=DEVICE))

# generate chinese poem with certain length
def generate(text,row,col):
    # define a recursive function to generate text with certain length
    def generate_loop(data):
        with torch.no_grad():
            out = model(**data)
        # Slices the logits tensor to retain only the logits corresponding to the last token in the sequence [batch_size, vocab_size]
        out = out["logits"][:,-1]
        # Retrieves the top-k (50) largest values along the last dimension [batch_size, 50]
        topk_value = torch.topk(out,50).values
        # Selects the last value in the top-k values tensor and add a new dimension so [batch_size, 1]
        topk_value = topk_value[:,-1].unsqueeze(dim=1)
    
        # Masks the logits tensor to retain only the logits that are greater than the last value in the top-k values tensor and set the rest to negative infinity
        out = out.masked_fill(out< topk_value,-float("inf"))
        out[:, tokenizer.get_vocab()["[UNK]"]] = -float("inf")
        for i in ",.()《》[]「」{}":
            out[:,tokenizer.get_vocab()[i]] = -float("inf")

        # Converts the logits tensor to a probability distribution tensor [batch_size, vocab_size] to avoid the model always generating the same token
        out = out.softmax(dim=1)
        # Samples a token ID from the probability distribution tensor [batch_size, 1]
        out = out.multinomial(num_samples=1)

        # Check if the generated token is a comma or period
        c = data["input_ids"].shape[1] / (col+1)
        if c %1 ==0:
            if c%2==0:
                #在偶数位添加句号
                out[:,0] = tokenizer.get_vocab()["."]
            else:
                #在奇数位添加逗号
                out[:,0] = tokenizer.get_vocab()[","]

        data["input_ids"] = torch.cat([data["input_ids"],out],dim=1)
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        data["token_type_ids"] = torch.ones_like(data["input_ids"]) # not needed for gpt2 actually
        data["labels"] = data["input_ids"].clone()

        if data["input_ids"].shape[1] >= row*col + row+1:
            return data
        # Recursively call the function to generate the next token
        return generate_loop(data)

    data = tokenizer.batch_encode_plus([text],return_tensors="pt")
    data["input_ids"] = data["input_ids"][:,:-1]
    data["attention_mask"] = torch.ones_like(data["input_ids"])
    data["token_type_ids"] = torch.ones_like(data["input_ids"])
    data["labels"] = data["input_ids"].clone()

    data = generate_loop(data)

    print(tokenizer.decode(data["input_ids"][i]))

if __name__ == '__main__':
    generate("白",row=4,col=5)