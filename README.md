# Caloscore v2 official repository

Official repository for CaloScore v2, an update to CaloScore that uses a diffusion generative model for fast detector simulation with single-shot sampling!


# Requirements

Packages used for training and sampling are found in the ```requirements.txt``` text file and can be directly installed with pip. 

# Data

Results are presented using the [Fast Calorimeter Data Challenge dataset](https://calochallenge.github.io/homepage/) and are available for download on zenodo:
* [Dataset 1](https://zenodo.org/record/6368338)
* [Dataset 2](https://zenodo.org/record/6366271)
* [Dataset 3](https://zenodo.org/record/6366324)

# Run the training scripts with

```bash
cd scripts
python train.py  --config CONFIG
```

* CONFIG options are ```[config_dataset1.json/config_dataset2.json/config_dataset3.json]```

After training the baseline model you can run the [progressive distillation](https://arxiv.org/abs/2202.00512) with the commands:

```bash
cd scripts
python train.py  --config CONFIG --distill --factor 2
```

For additional distillation steps just multiply ```--factor``` by a power of 2.



# Sampling from the learned score function

```bash
python plot_caloscore.py  --sample  --config CONFIG [--distill] [--factor 2]
```

Where again factor and distill flags are used to load the distilled models instead.

# Creating the plots shown in the paper

```bash
python plot_caloscore.py  --config CONFIG [--distill] [--factor 2]
```

A folder named ```plots``` are then going to be created with the results.


