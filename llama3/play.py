from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'we are at device {str(DEVICE)}')
MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype="auto",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

prompt ="hello, please introduce yourself."
message = [{"role":"system","content":"You are a helpful assistant system"},{"role":"user","content":prompt}]
text = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
model_inputs = tokenizer([text],return_tensors="pt").to(DEVICE)

generate_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)
response = tokenizer.batch_decode(generate_ids,skip_special_tokens=True)
print(response)

