# IMF-DDI
**Data preparation**

An innovative (Drug-Drug interaction)DDI predic- tion framework 
KGs information can be obtained from the following link：
1.https://het.io/
2.https://github.com/OpenBioLink/OpenBioLink

External entity relationships corresponding to drugs can be obtained by applying for a database file from the official website of drugbank.
https://go.drugbank.com/releases/latest


**Setting up the conda environment**

Use the command below to directly create an environment named IMFDDI.
```
conda env create -f environment.yaml
```

**Data processing**

Two ```.pkl``` files are required for training. One stores drug–drug interaction (DDI) information, while the other stores drug–external entity interaction information extracted from knowledge graphs (KGs). The following sections describe how to generate these two files from the original raw data.

1.First, use the following command to obtain the pkl file for drug interactions.
```
cd Drugbank
```
```
python utils/drugbank_pkl.py \
  --train data/train.csv \
  --val data/val.csv \
  --test data/test.csv \
  --ddi data/drugbank_ddi_id.txt \
  --output pkl/drugbank_event.pkl
```
2.We need to extract external entity information related to drugs involved in training from the downloaded KGs.
We attempted to extract external entity information from multiple KG files using Drugbank ID and Pubchem ID, and supplemented it with the Drugbank database file. For the convenience of reproducing the process described in the article, we directly provide the processed ```.csv``` file(Drugbank/data/drugbank.csv). If you need to introduce other KGs, you can supplement the information according to the format of thi
```
python utils/generate_node_representation.py \
  --csv data/drugbank_all_pathwayv2.csv \
  --txt data/embedding/drugbank_all_part3.txt \
  --output pkl/drugbank_pathway.pkl \
  --full
```
