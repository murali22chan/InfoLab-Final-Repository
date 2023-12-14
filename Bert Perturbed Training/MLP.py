import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import transformers

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')


# -------------------------------------------------------------------------------------------------------------------------------

# ##### Function to Span Masking

buffer_size = 0 # buffer size used to set space between two span masks


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2) #Number of times the span operation should occur for given probability.
    
    if ceil_pct:
        n_spans = np.ceil(n_spans)
        
    n_spans = int(n_spans) 

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length) #Randomly choose tokens with span length
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text




sample_text = "Philosophy does not promise to secure anything external for man, otherwise it would be admitting something that lies beyond its proper subject-matter. For as the material of the carpenter is wood, and that of statuary bronze, so the subject-matter of the art of living is each person's own life."

# tokenize_and_mask(sample_text, 2, 0.15)


# -------------------------------------------------------------------------------------------------------------------------------

# #### Function to Mask Filling

import re
pattern = re.compile(r"<extra_id_\d+>")


# In[52]:


def count_masks(text):
    count = 0
    for x in text.split():
        if x.startswith("<extra_id_"):
            count+=1
    return count




def extract_fills(text):
    # remove <pad> from beginning of each text
    text = text.replace("<pad>", "").replace("</s>", "").strip()

    # return the text in between each matched mask token
    extracted_fills = pattern.split(text)[1:-1]
    
    # remove whitespace around each fill
    extracted_fills = [y.strip() for y in extracted_fills]

    return extracted_fills




# Function to apply the extracted fills

def apply_extracted_fills(masked_text, extracted_fills):
    
    tokens = masked_text.split(" ")
    
    n = count_masks(masked_text)
    
    for idx in range(n):
        tokens[tokens.index(f"<extra_id_{idx}>")] = extracted_fills[idx]

    # join tokens back into text
    
    perturbed_text = " ".join(tokens)
    
    return perturbed_text
        
    
def tokens_length(data):
    tokenizer_func = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512)
    tokenized_data = tokenizer_func(data)
    return len(tokenized_data['input_ids'])



def fill_mask(model, tokenizer, masked_text):
    
    model.to('cuda')
    input_ids = tokenizer.encode(masked_text, padding = True, return_tensors = 'pt')
    
    n_expected = count_masks(masked_text)
    stop_id = tokenizer.encode(f"<extra_id_{n_expected}>")[0]
    
    input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model.generate(input_ids = input_ids, max_length = 512, do_sample = True, num_return_sequences = 1, top_p = 0.96, eos_token_id=stop_id)
    mask_filled_text = tokenizer.decode(outputs[0], skip_special_tokens= False)
    
    return mask_filled_text


# Perturbation Pipeline

# sample_masked_text = tokenize_and_mask(sample_text, 2, 0.15)
# model_masked_filled_output = fill_mask(model, tokenizer, sample_masked_text)
# extracted_fills = extract_fills(model_masked_filled_output)
# pert_text = apply_extracted_fills(sample_masked_text, extracted_fills)

# print("Original Text: ",sample_text)
# print("\n")
# print("Perturbed Text: ",pert_text)

# -------------------------------------------------------------------------------------------------------------------------------




