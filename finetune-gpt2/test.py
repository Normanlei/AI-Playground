import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'we are at device {str(DEVICE)}')
MODEL_PARAM_PATH = f"../trained_models/gpt2-chinese-4.pt"  # Path to the saved model parameters
MODEL_PATH = '../model/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3'


model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Load model and weights
model = model.to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PARAM_PATH, map_location=DEVICE))
model.eval()

# Define a few test examples
test_examples = [
    "床前明月光，",
    "天上的星星，",
    "春风又绿江南岸，",
    "白日依山尽，"
]

for example in test_examples:
    inputs = tokenizer.encode(example, return_tensors='pt').to(DEVICE)
    
    # Generate text
    output = model.generate(
        inputs,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"Generated: {generated_text}")
