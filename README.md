# Continual Cross-Dataset Adaptation in Road Surface Classification

Accurate road surface classification is crucial for autonomous vehicles (AVs) to optimize driving conditions, enhance safety, and enable advanced road mapping. 
However, deep learning models for road surface classification suffer from poor generalization when tested on unseen datasets. 
To update these models with new information, also the original training dataset must be taken into account, in order to avoid catastrophic forgetting. 
This is, however, inefficient if not impossible, e.g., when the data is collected in streams or large amounts. 
To overcome this limitation and enable fast and efficient cross-dataset adaptation, we propose to employ continual learning finetuning methods designed to retain past knowledge 
while adapting to new data, thus effectively avoiding forgetting. Experimental results demonstrate the superiority of this approach over naive finetuning, 
achieving performance close to fresh retraining. While solving this known problem, we also provide a general description of how the same technique can be adopted in other AV scenarios. 
We highlight the potential computational and economic benefits that a continual-based adaptation can bring to the AV industry, while also reducing greenhouse emissions due to unnecessary joint retraining.

## Run

```
main_experiments.py

positional arguments:
  strategy_name         Strategy name: 'lfl' | 'lwf' | 'naive' | 'cumulative' | 'joint'

optional arguments:
  -h, --help            show this help message and exit
  --dsorder DSORDER DSORDER DSORDER
                        Order of datasets e.g. [0,1,2], where 0=RTK, 1=KITTI, 2=CaRINA
  --perm {0,1,2,3,4,5}  Permutation number of datasets. 0=[0,1,2], ..., 5=[2,1,0]
  -l LOOP, --loop LOOP  How many times to loop through datasets (default 1)
  -w, --wandb           Enable wandb logger
  -v, --verbose         Enable interactive logger
```

### Project structure

```
.
├── data/                   # these datasets are already public; please reach out for further details.
│   ├── enlarged_dataset_CaRINA/
│   │   ├── asphalt/
│   │   ├── paved/
│   │   └── unpaved/
│   ├── enlarged_dataset_KITTI/
│   │   ├── asphalt/
│   │   └── paved/
│   └── enlarged_dataset_RTK/
│       ├── asphalt/
│       ├── paved/
│       └── unpaved/
├── dataloaders/
├── docker/
├── models/
├── outputs/
├── utils/
├── sklearn_metrics.py
├── torchmetrics_metrics.py
└── main_experiments.py     # main script
```

## Cite us

If you use this work, please cite us!

**Continual Cross-Dataset Adaptation in Road Surface Classification** [[arXiv](https://arxiv.org/abs/2309.02210)]<br>
P. Cudrano, M. Bellusci, G. Macino, and M. Matteucci<br>
Presented at 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC), Sep 2023

```
@misc{cudrano2023continual,
      title={Continual Cross-Dataset Adaptation in Road Surface Classification}, 
      author={Paolo Cudrano and Matteo Bellusci and Giuseppe Macino and Matteo Matteucci},
      year={2023},
      eprint={2309.02210},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
