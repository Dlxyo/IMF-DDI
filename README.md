# IMF-DDI

"IMF-DDI: Information mapping and fusion framework for drug-drug interaction prediction" has been accepted for publication in *Interdisciplinary Sciences: Computational Life Sciences*.

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

1.1 First, use the following command to obtain the pkl file for drug interactions.
```
cd Drugbank
```
```
python utils/drugbank_event_pkl.py \
  --train data/train.csv \
  --val data/val.csv \
  --test data/test.csv \
  --ddi data/drugbank_ddi_id.txt \
  --output pkl/drugbank_event.pkl
```
1.2 We need to extract external entity information related to drugs involved in training from the downloaded KGs.
We attempted to extract external entity information from multiple KG files using Drugbank ID and Pubchem ID, and supplemented it with the Drugbank database file. For the convenience of reproducing the process described in the article, we directly provide the processed ```.csv``` file(Drugbank/data/drugbank.csv). If you need to introduce other KGs, you can supplement the information according to the format of this. All initial representations used in this project are from unimol. You can choose to replace them with other molecular representation models that represent three-dimensional information.
```
python utils/drugbank_drug_pkl.py \
  --csv data/drugbank.csv \
  --txt data/unimol_repr_drugbank.txt \
  --output pkl/drugbank_drug.pkl \
  --four_entity
```
1.3 Before executing the training file, you need to manually change the pkl file path to the generated path.
```
python train.py
```
2.1 Same as Drugbank to process TWOSIDES, the data files is too large and needs to be manually decompressed from the compressed package and placed in the data folder. The pkl file is too large and exceeds the upload limit. Please generate it according to the steps.
```
cd TWOSIDES
```
```
python utils/TWOSIDES_event_pkl.py \
  --train_csv data/train_pol.csv \
  --val_csv data/valid_pol.csv \
  --test_csv data/test_pol.csv \
  --output_file pkl/TWOSIDES_event.pkl \
  [--max_ddi 200]
```

2.2 Same as Drugbank to process TWOSIDES
```
python utils/TWOSIDES_drug_pkl.py \
  --txt_file3 data/unimol_repr_TWOSIDES.txt \
  --csv_file data/TWOSIDES.csv \
  --output_file pkl/TWOSIDES_drug.pkl \
  --four_entity
```
1.3 Before executing the training file, you need to manually change the pkl file path to the generated path.
```
python train.py
```
