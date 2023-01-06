# Teach and Explore: A MultiplexInformation-guided Effective and Efficient Reinforcement Learning for Sequential Recommendation 
This repository contains the code for our paper Teach and Explore: A Multiplex Information-guided Effective and Efficient Reinforcement Learning for Sequential Recomemndation (MELOD0).
## Requirements
* Install Python, Pytorch. We use Python 3.8, Pytorch 1.11.0
* numpy (version: 1.22.3).
* tqdm (version: 4.64.0).
## Datasets
In our experiments, the Beauty, Cell Phone, Cloth, CD, Grocery and Toys are from http://jmcauley.ucsd.edu/data/amazon/, the Yelp dataset is from https://www.yelp.com/dataset.
## Knowledge Graph Embedding
The knowledge embedding is trained on [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
## Quick Start
You can run the model with the following code:
"""
python ../runMELOD.py --dataset toys --alpha 0.1  --lamda 0.5 --sas_prob 3
"""

