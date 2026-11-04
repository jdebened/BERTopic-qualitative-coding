# BERTopic-qualitative-coding

## Installation

Localize the following documents/libraries and put them in correspoding folders (structure below):
- [news_dialogue.json](https://github.com/zcgzcgzcg1/MediaSum)
- [20news_groups](https://www.kaggle.com/datasets/crawford/20-newsgroups)
- [BERTopic](https://github.com/MaartenGr/BERTopic)
- [flan-t5-base](https://huggingface.co/google/flan-t5-base)
- [Llama-2-2b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [transformers](https://github.com/huggingface/transformers)
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (after first clone may be in `sentence_transformers` folder but have to move to `sentence-transformers` folder)

```md
/root
│
├── /20news_groups
│
├── /BERTopic
│
├── /Google
│   ├── /flan-t5-base
│
├── /llama2
│   ├── /Llama-2-2b-hf
│
├── news_dialogue.json
│
├── /sentence-transformers
│   ├── /all-MiniLM-L6-v2
│
├── /sentence_transformers
│
├── /transformers
│
├── *.sh, .py files
├── README.md
└── .gitignore
```

## Running 
0. Trial (sample of 80 docs embedded in Python code):
  - run-trial-llama.sh: Llama 2 label
    
1. 20_newsgroups:
  - run-20groups-llama.sh: Llama 2 label
  - run-20groups-llm.sh: Flan-T5-base label
  - run-20groups-semisup_llmlabel.sh: Semisupervised, then Flan-T5-base label
  - run-20groups-semisup.sh: Semisupervised
    
2. news_dialogue.json (outdated):
Please execute run-content-summary.sh first to group (got categories for later semisupervised training)
  - run-content-llama.sh: Llama2 label
  - run-content-llm.sh: Flan-T5-base label
  - run-content-semisup_llmlabel.sh: Semisupervised, then Flan-T5-base label
  - run-content-seedwords.sh: Seed words (editable with different seed words in this file)
  - run-content-semisup.sh: Semisupervised

3. news_dialogue.json (final):
Please execute run-contentF-summary.sh first to group (got categories for later semisupervised training). Run metrics_content.py to compare results (need to modify the folder of saved artefacts).
  - run-contentF-llama_variable.sh, run-contentF-llama_variable_outlier.sh: Llama2 label + Variable number of topics + Without and with outlier reduction
  - run-contentF-llama_fixed.sh, run-contentF-llama_fixed_outlier.sh: Llama2 label + Fixed number of topics (n = 5, 10, 15, 20, 25, 30, 40, 50) + Without and with outlier reduction
  - run-contentF-seedwords_modified.sh: Seed words (editable with different seed words in this file) + Fixed number of topics (n = 15) + Without outlier reduction
  - run-contentF-semisup_modified.sh: Semisupervised + Fixed number of topics (n = 15) + Without outlier reduction

4. MTS dataset:
Please execute `python mts_preprocess.py` first to create `mts_dialog_merged.json`, then submit one of the following jobs:
  - `sbatch mts-run-llama_fixed_outlier.sh`: Llama2 label + Fixed number of topics (n = 5, 10, 15, 20, 25)
  - `sbatch mts-run-llama_variable_outlier.sh`: Llama2 label + Variable number of topics
