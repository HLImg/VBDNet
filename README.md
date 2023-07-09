# Variational Bayesian deep network for blind Poisson denoising

**The training model parameter files can be downloaded [here (google drive)](https://drive.google.com/drive/folders/1X8mlLL5_8xB5sYmv9N2KpGpa2R6dYGve?usp=sharing).**

All methods are trained using the same synthetic dataset and tested on synthetic and real-world test sets. All methods use early stopping techniques.

## how to test

```python
python main.py --yaml yaml/test_fmd/*.yaml
```

```bib
@article{liang2023variational,
  title={Variational Bayesian deep network for blind Poisson denoising},
  author={Liang, Hao and Liu, Rui and Wang, Zhongyuan and Ma, Jiayi and Tian, Xin},
  volume = {143},
  pages={109810},
  year={2023},
  journal={Pattern Recognition},
  doi = {https://doi.org/10.1016/j.patcog.2023.109810},
  publisher={Elsevier}
}


```
