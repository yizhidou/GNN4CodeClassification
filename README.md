### Datasets

- **Juliet Test Suite C/C++ (V1.3)**
  - The orignal dataset was available at. But the site seems to have changed a lot since we downloaded data from it (2021-6-11). To avoid ambiguity, we provide the exact version we emploied in our paper here (`GNN4Code/juliet_testsuite_origin_from_website.zip`): [https://drive.google.com/drive/folders/1_3BbBpi3dr2Q1SKQPruY9NvvikIaTped?usp=sharing] 
  - **Juliet-multi**
    - CWE-no to label mapping is in `./label_mappings/juliet_multi_label_mapping.json`
  - **Juliet-binary**
    - CWE-no to label mapping is in `./label_mappings/juliet_binary_label_mapping.json`
  - **Juliet-few**
    - All labels in Juliet-binary but not in Juliet-multi are in Juliet-few :). Or, put it another way, all classes that posses less than 1,000 samples are included in Juliet-few. Label distribution details are in `./auxiliary_records/juliet_cwe_label_distribution.json`
- **ReVeal**
  - The original paper:[https://arxiv.org/abs/2009.07235]
- **Devign**
  - The original paper:[https://arxiv.org/abs/1909.03496]
- **POJ-104**
  - The original paper:[https://arxiv.org/abs/1409.5718]

### Raw Code to CPG

- We use the open source tool Joern (https://joern.io/). The exact version we use is included in `./joern`

### Data Initialization

- Every node/edge in a CPG is preserved with its node/edge type only (in one-hot encoding). We did so because the naming convention of some datasets (the ones from Juliet Test Suite) contains label leakage, which should be avoided during experiments.  `.nm` indicates a node matrix file while `.am` indicates an adjacent node matrix. All files are named as their hash digits. All datasets in its features (.npy format) is available at.

### Dataset Division

- Cross division in RQ1, train/validation/test division in RQ2, RQ3, and RQ4 is all provided under `./division` .

### Radom Seed

- For 5-fold cross validation, we used `rand_seed=6`; for Three repeated experiments in RQ2 and RQ4, we used `rand_seed=6,8,10`.

### Acknowledgement

- Model Implementation was based on this project [https://github.com/Lin-Yijie/Graph-Matching-Networks]
