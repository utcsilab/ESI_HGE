# Hyperbolic Graph Embedding with Enhanced Semi-Implicit Variational Inference

### Requirements
- NumPy
- PyTorch 1.6 
- Scipy
- networkx
- json


```pip install -r requirements.txt``` 

### Installing

1. Clone the repository:
    ```shell
    $ git clone https://github.com/AliLotfi92/ESI_HGE
    $ cd esihge
    ```
2. Install requirements:
    ```shell
    $ pip install -r requirements.txt
    ```
---


#### Arguments:
* ```lr```: learning rate for the inference network
* ```dropout```: Dropout rate (1 - keep probability).
* ```epochs```: number of epochs to train the model. 
* ```c```: constant  negative  curvature
* ```K```: semi-implicit vi hyperparameters
* ```J```: semi-implicit vi hyperparameters
* ```dataset-str```: synthetic, cora, citeseer, or pubmed
