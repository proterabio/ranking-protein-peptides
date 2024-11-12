# Ranking Protein Peptides binding affinity

This code focuses on ranking the binding affinity between a given protein of interest and a list of peptides.

## Installation

The project requires the `transformers` library to calculate ESM2 embeddings. You will need to update the path to the ESM2 model weights in `src/utils.py` to match the location on your local system.

```bash
pip install transformers
```

## Usage
5 detailed case studies were provided: 
1_Example_HTRA1.ipynb
2_Example_HTRA3.ipynb
3_Example_MDM2.ipynb
4_Example_MDMX.ipynb
5_Example_EPIX4.ipynb
