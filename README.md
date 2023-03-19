# Variational Bayesian deep network for blind Poisson denoising

**The training model parameter files can be downloaded [here (google drive)](https://drive.google.com/drive/folders/1X8mlLL5_8xB5sYmv9N2KpGpa2R6dYGve?usp=sharing).**

All methods are trained using the same synthetic dataset and tested on synthetic and real-world test sets. All methods use early stopping techniques.

## how to test

```python
python main.py --yaml yaml/test_fmd/*.yaml
```
