from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd

#change this path to your local ESM checkpoints
torch.hub.set_dir("/data/ESM_checkpoints/")

#Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_path = "facebook/esm2_t33_650M_UR50D"
model = AutoModelForMaskedLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = model.to(device)
model.eval()



def compute_mlm_overall(protein, binder, protein_mask_positions, binder_mask_positions):
    protein_mask_positions = list(protein_mask_positions)
    binder_mask_positions = list(binder_mask_positions)
    
    # Concatenate protein sequences with a separator
    concatenated_sequence = protein + ":" + binder
    tokens = list(concatenated_sequence)

    # Create mask positions for protein and binder
    # Adjust binder positions to account for protein length and separator
    adjusted_binder_mask_positions = [pos + len(protein) + 1 for pos in binder_mask_positions]
    total_mask_positions = protein_mask_positions + adjusted_binder_mask_positions
    # Apply the mask token to the specified positions
    for pos in total_mask_positions:
        tokens[pos] = tokenizer.mask_token

    masked_sequence = "".join(tokens)
    inputs = tokenizer(masked_sequence, return_tensors="pt", truncation=True, max_length=1024, padding='max_length')
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the device


    # Compute the MLM loss
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    return loss.item()




def compute_mlm_overall_withoutTheMaskedPos(protein, binder, protein_mask_positions, binder_mask_positions):
    protein_mask_positions = list(protein_mask_positions)
    binder_mask_positions = list(binder_mask_positions)
    
    # Concatenate protein sequences with a separator
    concatenated_sequence = protein + ":" + binder
    tokens = list(concatenated_sequence)

    # Create mask positions for protein and binder
    # Adjust binder positions to account for protein length and separator
    adjusted_binder_mask_positions = [pos + len(protein) + 1 for pos in binder_mask_positions]
    total_mask_positions = protein_mask_positions + adjusted_binder_mask_positions
    # Apply the mask token to the specified positions
    for pos in total_mask_positions:
        tokens[pos] = tokenizer.mask_token

    masked_sequence = "".join(tokens)
    inputs = tokenizer(masked_sequence, return_tensors="pt", truncation=True, max_length=1024, padding='max_length')
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the device

    #Exclude masked positions from loss calculation
    inputs_temp = tokenizer(concatenated_sequence, return_tensors="pt", truncation=True, max_length=1024, padding='max_length')
    labels_c = inputs_temp["input_ids"].clone()
    total_mask_positions = [x + 1 for x in total_mask_positions] 
    mask = torch.zeros_like(labels_c, dtype=torch.bool)  # Initialize mask with False
    mask[:, total_mask_positions] = True # everything that is not on the masked postions should be taken into considreation in the loss
    labels_c[mask] = -100

    # Compute the MLM loss
    with torch.no_grad():
        outputs = model(**inputs, labels=labels_c)
        loss = outputs.loss

    return loss.item()



def find_differences(original, other):
    # Check that sequences are of the same length
    if len(original) != len(other):
        raise ValueError("Sequences must be of the same length.")

    # Identify the indices where the sequences differ hg
    differences = [i for i, (o, n) in enumerate(zip(original, other), start=0) if o != n]
    return differences