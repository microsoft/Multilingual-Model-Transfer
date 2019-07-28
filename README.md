# Zero-Resource Multilingual Model Transfer

This repo contains the source code for our ACL 2019 paper:

[**Multi-Source Cross-Lingual Model Transfer: Learning What to Share**](https://www.aclweb.org/anthology/P19-1299)
<br>
[Xilun Chen](http://www.cs.cornell.edu/~xlchen/),
[Ahmed Hassan Awadallah](https://www.microsoft.com/en-us/research/people/hassanam/),
[Hany Hassan](https://www.microsoft.com/en-us/research/people/hanyh/),
[Wei Wang](https://www.microsoft.com/en-us/research/people/wawe/),
[Claire Cardie](http://www.cs.cornell.edu/home/cardie/)
<br>
The 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)
<br>
[paper](https://www.aclweb.org/anthology/P19-1299),
[arXiv](https://arxiv.org/abs/1810.03552),
[bibtex](https://aclweb.org/anthology/papers/P/P19/P19-1299.bib)

## Introduction
Modern NLP applications have enjoyed a great boost utilizing neural networks models. Such deep neural models, however, are not applicable to most human languages due to the lack of annotated training data for various NLP tasks. Cross-lingual transfer learning (CLTL) is a viable method for building NLP models for a low-resource target language by leveraging labeled data from other (source) languages. In this work, we focus on the multilingual transfer setting where training data in multiple source languages is leveraged to further boost target language performance.

Unlike most existing methods that rely only on language-invariant features for CLTL, our approach coherently utilizes both **language-invariant** and **language-specific** features at instance level. Our model leverages adversarial networks to learn language-invariant features, and mixture-of-experts models to dynamically exploit the similarity between the target language and each individual source language. This enables our model to learn effectively what to share between various languages in the multilingual setup. Moreover, when coupled with unsupervised multilingual embeddings, our model can operate in a **zero-resource** setting where neither **target language training data** nor **cross-lingual resources** (e.g. parallel corpora or Machine Translation systems) are available. Our model achieves significant performance gains over prior art, as shown in an extensive set of experiments over multiple text classification and sequence tagging tasks including a large-scale industry dataset.

## Requirements
- Python 3.6
- PyTorch 0.4
- PyTorchNet (for confusion matrix)
- tqdm (for progress bar)


## File Structure
```
.
├── LICENSE
├── README.md
├── conlleval.pl                            (official CoNLL evaluation script)
├── data_prep                               (data processing scripts)
│   ├── bio_dataset.py                      (processing the CoNLL dataset)
│   └── multi_lingual_amazon.py             (processing the Amazon Review dataset)
├── data_processing_scripts                 (auxiliary scripts for dataset pre-processing)
│   └── amazon
│       ├── pickle_dataset.py
│       └── process_dataset.py
├── layers.py                               (lower-level helper modules)
├── models.py                               (higher-level modules)
├── options.py                              (hyper-parameters aka. all the knobs you may want to turn)
├── scripts                                 (scripts for training and evaluating the models)
│   ├── get_overall_perf_amazon.py          (evaluation script for Amazon Reviews)
│   ├── get_overall_perf_ner.py             (evaluation script for CoNLL NER)
│   ├── train_amazon_3to1.sh                (training script for Amazon Reviews)
│   └── train_conll_ner_3to1.sh             (training script for CoNLL NER)
├── train_cls_man_moe.py                    (training code for text classification)
├── train_tagging_man_moe.py                (training code for sequence tagging)
├── utils.py                                (helper functions)
└── vocab.py                                (building the vocabulary)
```


## Dataset
The CoNLL [2002](https://www.clips.uantwerpen.be/conll2002/ner/), [2003](https://www.clips.uantwerpen.be/conll2003/ner/) and [Amazon](https://webis.de/data/webis-cls-10.html) datasets, as well as the multilingual word embeddings ([MUSE](https://github.com/facebookresearch/MUSE), [VecMap](https://github.com/artetxem/vecmap), [UMWE](https://github.com/ccsasuke/umwe)) are all publicly available online.

Our pre-processed version may be provided separately to run our code with 1-click.

## Run Experiments

### CoNLL Named Entity Recogintion
```bash
./scripts/train_conll_ner_3to1.sh {exp_name}
```

The following script can print out a compiled dev/test F1 scores for all languages:

```bash
python scripts/get_overall_perf_ner.py save {exp_name}
```

### Multilingual Amazon Reviews
```bash
./scripts/train_amazon_3to1.sh {exp_name}
```

The following script can print out a compiled dev/test F1 scores for all languages:

```bash
python scripts/get_overall_perf_amazon.py save {exp_name}
```

## Citation

If you find this project useful for your research, please kindly cite our ACL 2019 paper:

```bibtex
@InProceedings{chen-etal-acl2019-multi-source,
    author = {Chen, Xilun and Hassan Awadallah, Ahmed and Hassan, Hany and Wang, Wei and Cardie, Claire},
    title = {Multi-Source Cross-Lingual Model Transfer: Learning What to Share},
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

