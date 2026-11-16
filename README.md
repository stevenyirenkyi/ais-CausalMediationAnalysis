# Reimplementing Causal Mediation Analysis for Gender Bias in GPT-2 with NNsight

This repository reimplements the [Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias](https://arxiv.org/pdf/2004.12265) using NNsight.

The four main things changed from the original implementation are:
1. Using `nnsight` for intervention
2. Using logit difference instead of probabilities for measuring effects
3. Conduction attention intervention with profession dataset instead of Winobias or Winogender

----

Intervention codes can be found in the `interventions` folder. The resulting data can be downloaded [here](https://drive.google.com/file/d/1WjK16ezwl-Hewe0bCl8B2CuFzCZ70oFp/view?usp=sharing). Once downloaded place the datasets in the `data` folder, like `data/interventions/attentions/heads/man_direct.npy`

---
`attn_analysis.ipynb` and `neuron_analysis.ipynb` contains the plots used to analyse the results.

---
Rough Draft can be found [here](https://docs.google.com/document/d/14HaaYnQqeKRC-UC03Y_mq03Ipwuo0EZsQ_ws65pacJg/edit?usp=sharing). Still working it. Need to settle on where to place it first
