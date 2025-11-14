# AI Basics — Teaching Notebooks

> A collection of Python/PyTorch notebooks covering supervised & unsupervised learning, CNNs, transformers/NLP, generative models, reinforcement learning, and LLM agents. Designed for **Colab-first** delivery with minimal setup.

## Quick Start

1. Click any **Open in Colab** badge below to launch the notebook in your browser.
2. GPU notebooks: in Colab choose **Runtime → Change runtime type → GPU**.

## Table of Contents

### Environment & Setup
- Linux / Docker / Conda basics (Markdown guides): [Basic_Coding_Environment_Command_Tutorial.md](./00%20Linux%20basic/Basic_Coding_Environment_Command_Tutorial.md)

### Python & PyTorch 

### NN Training & Optimization
- Optimizers & LR schedules (SGD, Adam, etc.): [Optimizer.ipynb](./02%20NN%20Training%20%26%20Optimization/Optimizer.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/02%20NN%20Training%20%26%20Optimization/Optimizer.ipynb)

### Supervised Learning
- Linear Regression from scratch: [01_Supervised_learning_Linear_regression_from_scratch.ipynb](./03%20Supervised%20Learning/linear_regression_from_scratch.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/03%20Supervised%20Learning/linear_regression_from_scratch.ipynb)
- Logistic Regression from scratch: [logistic_regression_from_scratch_ipynb.ipynb](./03%20Supervised%20Learning/logistic_regression_from_scratch.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/03%20Supervised%20Learning/logistic_regression_from_scratch.ipynb)

### Unsupervised Learning
- k-means clustering: [kmeans.ipynb](./04%20Unsupervised%20Learning/kmeans.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/04%20Unsupervised%20Learning/kmeans.ipynb)
- Principal Component Analysis (PCA): [pca.ipynb](./04%20Unsupervised%20Learning/pca.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/04%20Unsupervised%20Learning/pca.ipynb)

### Deep Neural Networks (DNN)
- Convolutional Neural Networks (CNN): [Building_a_CNN.ipynb](./05%20DNN/Building_a_CNN.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/05%20DNN/Building_a_CNN.ipynb)

### Transformers & NLP
- BERT Fine-tuning (Sentence Classification): [BERT_Fine_Tuning_Sentence_Classification.ipynb](./06%20Transformer/BERT_Fine_Tuning_Sentence_Classification.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/06%20Transformer/BERT_Fine_Tuning_Sentence_Classification.ipynb)
- GPT-2 : [ProtGPT2.ipynb](./06%20Transformer/ProtGPT2.ipynb)• [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/GPT2.ipynb)  

### Deep Generative Models
- Diffusion Models (intro): [Diffusion.ipynb](./07%20Deep%20Generative%20Models/Diffusion.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/07%20Deep%20Generative%20Models/Diffusion.ipynb)

### Reinforcement Learning
- Q-Learning (tabular / basics): [DQN.ipynb](./08%20Deep%20Reinforcement%20Learning/DQN.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/08%20Deep%20Reinforcement%20Learning/DQN.ipynb)

### LLM & Agents
- SFT with (Q)LoRA / TRL: [sft_trl_lora_qlora.ipynb](./09%20LLM%20%26%20Agent/sft_trl_lora_qlora.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/09%20LLM%20%26%20Agent/sft_trl_lora_qlora.ipynb)
- LangChain Agents (tools & reasoning): [langchain_agents.ipynb](./09%20LLM%20%26%20Agent/langchain_agents.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/09%20LLM%20%26%20Agent/langchain_agents.ipynb)

### Visualization
- Visualization Demos: [visualization.ipynb](./10%20Visualization/visualization.ipynb) • [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AI-HPC-Research-Team/AI-Basics/blob/main/10%20Visualization/visualization.ipynb)


## References & Credits

We gratefully acknowledge the following courses, tutorials, and repositories that inspired or were adapted in parts of these notebooks. Please cite them when reusing the corresponding materials:

- **taldatech/ee046211-deep-learning** — Jupyter Notebook tutorials for the Technion ECE 046211 Deep Learning course.  
  GitHub: https://github.com/taldatech/ee046211-deep-learning

- **yandexdataschool/mlhep2019** — MLHEP'19 slides and notebooks.  
  GitHub: https://github.com/yandexdataschool/mlhep2019

- **pmuens/lab** — Research environment to play with algorithms and data structures.  
  GitHub: https://github.com/pmuens/lab

- **ageron/handson-ml3** — Hands-On Machine Learning (3rd ed.) notebooks (scikit-learn, Keras, TensorFlow 2).  
  GitHub: https://github.com/ageron/handson-ml3

- **ImperialCollegeLondon/RCDS-Deep-Learning-CNN** — CNN tutorials from Imperial College London.  
  GitHub: https://github.com/ImperialCollegeLondon/RCDS-Deep-Learning-CNN

- **BERT Fine-Tuning Tutorial (PyTorch)** — Chris McCormick & Nick Ryan (2019-07-22).  
  Website: http://www.mccormickml.com

- **Multiomics-Analytics-Group/course_protein_language_modeling** — Course materials on protein language modeling.  
  GitHub: https://github.com/Multiomics-Analytics-Group/course_protein_language_modeling

- **huggingface/diffusion-models-class** — Materials for the Hugging Face Diffusion Models Course.  
  GitHub: https://github.com/huggingface/diffusion-models-class

- **pytorch/tutorials** (gh-pages) — Official PyTorch tutorials.  
  GitHub: https://github.com/pytorch/tutorials

- **huggingface/trl** — Train transformer language models with reinforcement learning.  
  GitHub: https://github.com/huggingface/trl

- **pinecone-io/examples** — Hands-on examples for Pinecone vector databases.  
  GitHub: https://github.com/pinecone-io/examples
