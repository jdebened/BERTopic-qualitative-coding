# BERTopic-qualitative-coding

## Installation

Localize the following documents/libraries and put them in correspoding folders (structure below):
- [news_dialogue.json](https://github.com/zcgzcgzcg1/MediaSum)
- [20news_groups](https://www.kaggle.com/datasets/crawford/20-newsgroups)
- [BERTopic](https://github.com/MaartenGr/BERTopic)
- [flan-t5-base](https://huggingface.co/google/flan-t5-base)
- [Llama-2-2b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [transformers](https://github.com/huggingface/transformers)
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)

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
  - run-trail-llama.sh: Llama 2 label
    
1. 20_newsgroups:
  - run-20groups-llama.sh: Llama 2 label
  - run-20groups-llm.sh or run-llm.sh: Flan-T5-base label
  - run-20groups-semisup_llmlabel.sh: Semisupervised, then Flan-T5-base label
  - run-semisup.sh: Semisupervised
    
2. news_dialogue.json:
Please execute run-summary.sh first to group (got categories for later semisupervised training)
  - run-content-llama.sh: Llama2 label
  - run-content-llm.sh or run-llm.sh: Flan-T5-base label
  - run-content-semisup_llmlabel.sh: Semisupervised, then Flan-T5-base label
  - run-content.sh: Semisupervised
  - run-seedwords_modfied.sh: Seed words (editable with different seed words in this file)

