# ZeroStance
This is the repository for our ACL2024 Findings paper: ZeroStance: Leveraging ChatGPT for Open-Domain Stance Detection via Dataset Generation

Please cite our paper at:
```
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
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.794",
    pages = "13390--13405",
    abstract = "Zero-shot stance detection that aims to detect the stance (typically against, favor, or neutral) towards unseen targets has attracted considerable attention. However, most previous studies only focus on targets from a single or limited text domains (e.g., financial domain), and thus zero-shot models cannot generalize well to unseen targets of diverse domains (e.g., political domain). In this paper, we consider a more realistic task, i.e., open-domain stance detection, which aims at training a model that is able to generalize well to unseen targets across multiple domains of interest. Particularly, we propose a novel dataset generation method ZeroStance, which leverages ChatGPT to construct a synthetic open-domain dataset CHATStance that covers a wide range of domains. We then train an open-domain model on our synthetic dataset after proper data filtering. Extensive results indicate that our model, when trained on this synthetic dataset, shows superior generalization to unseen targets of diverse domains over baselines on most benchmarks. Our method requires only a task description in the form of a prompt and is much more cost-effective and data-efficient than previous methods. We will release our code and data to facilitate future research.",
}
```
