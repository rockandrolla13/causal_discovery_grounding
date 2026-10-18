# ManyLabs1 raw data

Experiments I and III, and so the ablations, start from the ManyLabs1 SPSS file. Download `CleanedDataset.sav` from <https://osf.io/wx7ck/> and save it in this directory:

```
osrct_benchmark_v1.0/data/CleanedDataset.sav
```

This is the default `--manylabs-sav` of `paper/reproduce_experiment1_manylabs.sh` and `paper/reproduce_experiment3_bayesian.sh`. Their `data` stage converts the file to `ManyLabs1/pre-process/Manylabs1_data.pkl`, which every later stage reads. The data files are not part of this repository.
