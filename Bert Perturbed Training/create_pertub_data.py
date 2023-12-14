# Main fuction of the detect gpt algorithm 

import pandas as pd
import numpy as np
import json

import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

import MLP as MLP 

#Model for maskfilling
mask_tokenizer = T5Tokenizer.from_pretrained('t5-small')
mask_model = T5ForConditionalGeneration.from_pretrained('t5-small')



data = []
for line in open('all.jsonl', 'r', encoding="utf8"):
	data.append(json.loads(line))



finance_data = pd.DataFrame(columns = ["Question", "Text", "LabelName", "Label", "Perturbed Text"])

i_count = 1


for i in data:

	print("Data Processing: "+str(i_count))
	i_count+=1

	for j in range(2):
		if j == 0:
			temp_list = list(i["human_answers"])

			for temp_text in temp_list:
				temp_question = i["question"]
				temp_label = 0

				# If the input sequence lenght is 512 drop the value. (Mask Filling model cannot handle text with more than 512 len)
				if MLP.tokens_length(temp_text) > 512:
					pert_text = ""
					df_temp = {'Question': temp_question, 'Text': temp_text, 'LabelName': 'Human Answer','Label':temp_label, 'Perturbed Text':pert_text}
					finance_data = finance_data.append(df_temp, ignore_index = True)
					continue

				#Pertubation Pipeline

				flag = 0 # Use to handle failed attempts to generate mask


				while flag==0:
					masked_text = MLP.tokenize_and_mask(temp_text, 2, 0.15)
					n_expected = MLP.count_masks(masked_text)

					model_masked_filled_output = MLP.fill_mask(mask_model, mask_tokenizer, masked_text)

					extracted_fills = MLP.extract_fills(model_masked_filled_output)
					n_actual = len(extracted_fills)

					if n_expected == n_actual:
						pert_text = MLP.apply_extracted_fills(masked_text, extracted_fills)
						flag = 1
					else:
						flag = 0


				df_temp = {'Question': temp_question, 'Text': temp_text, 'LabelName': 'Human Answer','Label':temp_label, 'Perturbed Text':pert_text}
				finance_data = finance_data.append(df_temp, ignore_index = True)
			
		if j == 1:
			temp_list = list(i["chatgpt_answers"])

			for temp_text in temp_list:
				temp_question = i["question"]
				temp_label = 1

				if MLP.tokens_length(temp_text) > 512:
					pert_text = ""
					df_temp = {'Question': temp_question, 'Text': temp_text, 'LabelName': 'ChatGPT Answer','Label':temp_label,'Perturbed Text':pert_text}
					finance_data = finance_data.append(df_temp, ignore_index = True)
					continue

				#Pertubation Pipeline

				flag = 0 # Use to handle failed attempts to generate mask


				while flag == 0:
					masked_text = MLP.tokenize_and_mask(temp_text, 2, 0.15)
					n_expected = MLP.count_masks(masked_text)
					model_masked_filled_output = MLP.fill_mask(mask_model, mask_tokenizer, masked_text)
					extracted_fills = MLP.extract_fills(model_masked_filled_output)
					n_actual = len(extracted_fills)

					if n_expected == n_actual:
						pert_text = MLP.apply_extracted_fills(masked_text, extracted_fills)
						flag = 1
					else:
						flag = 0

				df_temp = {'Question': temp_question, 'Text': temp_text, 'LabelName': 'ChatGPT Answer','Label':temp_label,'Perturbed Text':pert_text}
				finance_data = finance_data.append(df_temp, ignore_index = True)

	if i_count == 200:
		break

finance_data.to_csv('Full_Data_With_Perturbed_Text.csv',index = False)