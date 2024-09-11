# ZeroStance
This is the repository for our ACL2024 Findings paper: ZeroStance: Leveraging ChatGPT for Open-Domain Stance Detection via Dataset Generation

## Abstract

Zero-shot stance detection that aims to detect the stance (typically against, favor, or neutral) towards unseen targets has attracted considerable attention. However, most previous studies only focus on targets from a single or limited text domains (e.g., financial domain), and thus zero-shot models cannot generalize well to unseen targets of diverse domains (e.g., political domain). In this paper, we consider a more realistic task, i.e., open-domain stance detection, which aims at training a model that is able to generalize well to unseen targets across multiple domains of interest. Particularly, we propose a novel dataset generation method ZeroStance, which leverages ChatGPT to construct a synthetic open-domain dataset CHATStance that covers a wide range of domains. We then train an open-domain model on our synthetic dataset after proper data filtering. Extensive results indicate that our model, when trained on this synthetic dataset, shows superior generalization to unseen targets of diverse domains over baselines on most benchmarks. Our method requires only a task description in the form of a prompt and is much more cost-effective and data-efficient than previous methods. We will release our code and data to facilitate future research.

## Running

Please check .sh file for main experiments, ablation study and the analysis. 

## Data

`chatgpt_carto_bertweet_var_0.99_seed0` indicates the final __CHATStance dataset__. If you only have an insterest in this dataset, please feel free to use it in your customized setting.

`chatgpt` indicates the raw CHATStance dataset without data filtering step.

`chatgpt_carto_var_0.99_1type_2_seed0` indicates the data used in ablation study (Table 5), e.g., only using prompt 2. Prompt IDs 2, 3, 5 refer to 1, 2, 3 in the paper.

`chatgpt_carto_var_0.99_2types_23_seed0` indicates the data used in ablation study (Table 5), e.g., not using prompt 5.

`chatgpt_carto_var_0.99_seed0_subset_500` indicates the data used in Table 6. 500 -> 16\%, 1000 -> 33\%, 2000 -> 66\%.

`chatgpt_*` indicates the data used in Table 8, e.g., chatgpt_vast. Each human-annotated dataset is used as the open-domain dataset.

`openstance_*` indicates the data used for OpenStance baseline in Table 4.

`{dataset}` indicates the existing dataset used for training (in the out-of-domain setting) and evaluation.

## Contact Info

Please contact Chenye Zhao (czhao43@uic.edu) or Yingjie Li (liyingjie@westlake.edu.cn) with any questions.

## Citation

```bibtex
@inproceedings{zhao-etal-2024-zerostance,
    title = "{Z}ero{S}tance: Leveraging {C}hat{GPT} for Open-Domain Stance Detection via Dataset Generation",
    author = "Zhao, Chenye  and
      Li, Yingjie  and
      Caragea, Cornelia  and
      Zhang, Yue",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    url = "https://aclanthology.org/2024.findings-acl.794",
    pages = "13390--13405",
}
```
