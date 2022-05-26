<h1 align="center">
Edge-featured Graph Neural Architecture Search
</h1>



## System Requirements
+ Linux
+ Python 3.6
+ Pytorch 1.9.1
+ DGL 0.6.1
+ CUDA toolkit 11.3
+ One NVIDIA GPU such as RTX 3090. 

Run the following command for building the environment. 
```sh
sudo apt install graphviz
conda env create -f environment.yml
conda activate gnas
```

## Dataset Preparation
Some datasets (CLUSTER, TSP, ZINC, and CIFAR10) are provided by project [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns). 
|DATASET|TYPE|URL|
|---|---|---|
|CLUSTER|node|[click here](https://data.dgl.ai/dataset/benchmarking-gnns/SBM_CLUSTER.pkl)|
|TSP|edge|[click here](https://data.dgl.ai/dataset/benchmarking-gnns/TSP.pkl)|
|ZINC|graph|[click here](https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl)|
|MNIST|graph|[click here](https://data.dgl.ai/dataset/benchmarking-gnns/MNIST.pkl)|
|CIFAR10|graph|[click here](https://data.dgl.ai/dataset/benchmarking-gnns/CIFAR10.pkl)|

## Search GNN Architectures

We have provided scripts for easily searching graph neural networks on six datasets. 
```sh
CUDA_VISIBLE_DEVICES=0 python search.py ds=ZINC optimizer=train_optimizer ds.arch_save='archs/TEST' basic.nb_layers=4 basic.nb_nodes=4
```

## Train with Genotypes
We provided scripts for easily training graph neural networks searched by ARGNP.
```sh 
CUDA_VISIBLE_DEVICES=0 python train.py ds=ZINC  optimizer=train_optimizer ds.load_genotypes='archs/TEST/ZINC/45/cell_geno.txt'
```
