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

1.use commond
```
python Drugbank/utils/drugbank_pkl.py \
  --train data/train.csv \
  --val data/val.csv \
  --test data/test.csv \
  --ddi data/drugbank_ddi_id.txt \
  --output data/drugbank_event.pkl

```
