# Description
EnGCI is a state-of-the-art computational tool designed for predicting the binding affinity between G protein-coupled receptors (GPCRs) and small-molecule compounds. This program enables users to efficiently evaluate GPCR-compound interactions using raw GPCR sequences and compound SMILES inputs, even for targets in unbound states.

To prioritize accessibility and user-friendliness, EnGCI is hosted on a dedicated cloud server, eliminating the need for local installation. The web-based platform supports seamless input (GPCR sequence and SMILES) and outputs affinity predictions through an automated workflow, including feature processing and model inference.Researchers can currently access the tool via a temporary public server at：127.0.0.1:7868

# Source codes:
- create_GPCR_data.py: Processing pytorch data for model 1
- create_data_esm.py: Processing ESM data for model 2
- creat_data_uni-mol.py: Processing uni-mol data for model 2
- esm_dict.py: Convert the data processed by the macromolecule model into a dictionary for use in model 2
- train_validation_GPCR.py: For training and validating model 1
- esm_uni_mol_train_validation.py: For training and validating model 2
- utils.py: include TestbedDataset used by create_GPCR_data.py to create data, and performance measures.
- esm_uni_mol_GIN_eval.py: For testing the ensemble model EnGCI
- ginconv_GPCR_new_3conv.py: This is model 1; gat_gcn.py, gat.py, gcn.py for comparison
- esm_uni_mol.py: This is model 2
- gin_t_sne.py: Model 1 visualization
- esm_t_sne.py: Model 2 visualization
- Integrated_model_visualization: Integrated model visualization
- only_GNN_eval: Model 1 test alone
- only_esm_eval: Model 2 test alone

# Step-by-step running:
##  1.Install Python libraries needed
- Install rdkit: conda install -y -c conda-forge rdkit
- Install uni-mol: pip install unimol_tools, pip install huggingface_hub
- Install esm: pip install fair-esm, esm needs to download esm2_t33_650M_UR50D.pt and esm2_t33_650M_UR50D-contact-regression.pt
```python
conda create -n engci python=3
conda activate engci
conda install -y -c conda-forge rdkit
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```
## 2.Create data for model 1 and model 2
Running
```python
create_GPCR_data.py
create_data_esm.py
esm_dict.py
creat_data_uni-mol.py
```
## 3.Train and verify model 1
To train a model using training data. The model is chosen if it gains the best MSE for testing data.
Running
```python
conda activate engci
python train_validation_GPCR.py 0 0 0
```
where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively; the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet, respectively; and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively.

## 4.Train and verify model 2
```python
python esm_uni_mol_train_validation.py
```
## 5.Test model 1, model 2 individually and test the integrated model
```python
python only_GNN_eval.py
python only_esm_eval
python esm_uni_mol_GIN_eval.py
```
