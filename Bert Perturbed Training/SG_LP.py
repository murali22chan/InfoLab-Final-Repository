import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
model = GPT2LMHeadModel.from_pretrained('gpt2') # Loss is KL diverenge, which is -log likelyhood
model.eval()


#### Function to get Σ log(Pθ)

# function to get Sigma log(p0)
def get_sigma_log_p0(model, tokenizer, text):
    model.to('cuda')
    input_ids = tokenizer.encode(text, return_tensors = 'pt')
    
    input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels = input_ids)
    loss = outputs[0]
    
    sentence_prob = loss.item()
    return -sentence_prob



# get_sigma_log_p0(model,tokenizer,"Friday thesis report")


# -------------------------------------------------------------------------------------------------------------------------------

#### Generating Samples Using GPT 2


def generate_mgt(model, tokenizer, text, max_length, sampling_type = "top k"):
    
    model.to('cuda')
    
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.to('cuda')
    
    if sampling_type == "top k":
        
        #Using Top K sampling to generate text k = 40
        with torch.no_grad():
            output = model.generate(input_ids = input_ids,max_length = max_length, do_sample = True, top_k = 40)  
    
    if sampling_type == "top p":
        #Using Top P sampling to generate text p = 0.96
        
        with torch.no_grad():
            output = model.generate(input_ids = input_ids,max_length = max_length, do_sample = True, top_k = 0, top_p = 0.96)
        
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response_text

# generate_mgt(model, tokenizer, "Friday thesis report", 10, sampling_type = "top p")






