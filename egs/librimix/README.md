
LibriMix是一个用于语音前端的数据集. LibriMix基于LibriSpeech构造, 生成方法详见[repo_librimix](https://github.com/ZhaZhaFon/librimix_config)  

egs/librimix包含以下几个前端模型的实现, 粗体表示已跑
* Conv-TasNet
* **DCCRN**
* DCUNet
* DPRNN
* DPTNet
* SuDORMRFImproved
* SuDORMRF

---
---

### LibriMix dataset

The LibriMix dataset is an open source dataset 
derived from LibriSpeech dataset. It's meant as an alternative and complement
to [WHAM](./../wham/).

More info [here](https://github.com/JorisCos/LibriMix).

**References**
```BibTeX
@misc{cosentino2020librimix,
    title={LibriMix: An Open-Source Dataset for Generalizable Speech Separation},
    author={Joris Cosentino and Manuel Pariente and Samuele Cornell and Antoine Deleforge and Emmanuel Vincent},
    year={2020},
    eprint={2005.11262},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
