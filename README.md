# Reimplementing Causal Mediation Analysis for Gender Bias in GPT-2 with NNsight

This repository reimplements the [Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias](https://arxiv.org/pdf/2004.12265) using NNsight.

The four main things that changed from the original implementation are:
1. Using `nnsight` for intervention
2. Using logit difference instead of probabilities for measuring effects
3. Running the attention head intervention on the profession dataset instead of Winobias/Winogender.

The main takeaways are:
1. No single neuron mediates gender bias, consistent with Vig et al. (2020).
2. Attention heads don’t carry most of the effects either, suggesting that Vig et al. (2020) findings do not generalize across different datasets and intervention setups.
3. Total effect is approximately equal to the sum of natural direct and indirect effects, consistent with Vig et al. (2020). It is tighter when professions already match the swapped-in gender.

----



Intervention codes can be found in the `interventions` folder. The resulting data can be downloaded [here](https://drive.google.com/file/d/1WjK16ezwl-Hewe0bCl8B2CuFzCZ70oFp/view?usp=sharing). Once downloaded, place the datasets in the `data` folder, like `data/interventions/attentions/heads/man_direct.npy`

---
`attn_analysis.ipynb` and `neuron_analysis.ipynb` contain the plots used to analyse the results.

---
A rough draft can be found [here](https://docs.google.com/document/d/14HaaYnQqeKRC-UC03Y_mq03Ipwuo0EZsQ_ws65pacJg/edit?usp=sharing). Still working on it. Need to settle on where to place it first
