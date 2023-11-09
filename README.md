# Skip Connections and Training Cost Reduction: A Game Changer in  Graph Neural Networks 

This repository contains a PyTorch implementation of "Skip Connections and Training Cost Reduction: A Game Changer in Graph Neural Networks".

## Dependencies
- CUDA 10.1
- python 3.6.9
- pytorch 1.3.1
- networkx 2.1
- scikit-learn

## Datasets

The `data` folder contains three benchmark datasets(Cora, Citeseer, Pubmed), and the `newdata` folder contains four datasets(Chameleon, Cornell, Texas, Wisconsin) from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn). We use the full-supervised setting as [GCN](https://github.com/tkipf/gcn).

## Results
Full supervised experimental results for mean classification accuracy and training cost for node classification
Improved results are shown in bold. (A full comparison is shown in the paper.)

| Dataset | Training Cost(sec) |  Accuracy | Dataset | Training Cost(sec) |  Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Cora       | **480.00** | 88.03  | Cham | **79.6865**  | **67.43** |
| Cite       | 269.95 | **76.99**  | Corn | **185.32** | **74.59** |
| Pubm       | **3168.37** | **67.43**  | Texa | 369.5642 | **79.46** |
| Wisc | **104.89** | 77.06 |



## Usage

To replicate the full-supervised results, run the following scripts:

```sh
python -u full-supervised.py --data cora --layer 64 --alpha 0.2 --weight_decay 1e-4
python -u full-supervised.py --data citeseer --layer 64 --weight_decay 5e-6
python -u full-supervised.py --data pubmed --layer 64 --alpha 0.1 --weight_decay 5e-6
python -u full-supervised.py --data chameleon --layer 8 --lamda 1.5 --alpha 0.2 --weight_decay 5e-4
python -u full-supervised.py --data cornell --layer 16 --lamda 1 --weight_decay 1e-3
python -u full-supervised.py --data texas --layer 32 --lamda 1.5 --weight_decay 1e-4
python -u full-supervised.py --data wisconsin --layer 16 --lamda 1 --weight_decay 5e-4
```
