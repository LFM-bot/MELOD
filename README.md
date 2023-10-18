# Teach and Explore: A Multiplex Information-guided Effective and Efficient Reinforcement Learning for Sequential Recommendation 
This repository contains the code for our paper Teach and Explore: A Multiplex Information-guided Effective and Efficient Reinforcement Learning for Sequential Recomemndation (MELOD).
## Requirements
* Install Python, Pytorch. We use Python 3.8, Pytorch 1.11.0
* numpy (version: 1.22.3)
* tqdm (version: 4.64.0)
## Overview
MELOD considers the SR task as a sequential decision problem and mainly consists of three novel extensions: state encoding, policy function, and RL training. The state encoding endows MELOD with the capability for learning a comprehensive user representation by encoding sequential and knowledge state representations. A dynamic intent induction network (DIIN) is designed as a policy function to conduct a sequential signal and semantic knowledge joint-guided prediction. The RL training framework uses Teach and Explore components to capture users’ explicit preferences and potential interests for SRs.
![Image text](https://github.com/LFM-bot/Teach-and-Explore-A-Multiplex-Information-guided-Effective-and-Efficient-RL-for-SR/blob/master/fig/model.png)
## Datasets
In our experiments, the Beauty, Cell Phone, Cloth, CD, Grocery and Toys datasets are from http://jmcauley.ucsd.edu/data/amazon/, the Yelp dataset is from https://www.yelp.com/dataset.
## Knowledge Graph Embedding
The knowledge embedding is trained on [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
## Quick Start
You can run the model with the following code:
```
python runMELOD.py --dataset toys --alpha 0.4  --lamda 0.2 --sas_prob 3
```


