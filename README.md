# SpatioTemporal BLIS-Net (STBLIS-Net)
## CPSC583 (Fall 2024) Final Project - Chaz Okada

### Installation instructions
1) Create a directory to clone the git repo
2) Create a conda environment using the provided `environment.yaml` file.
3) Activate the environrment
4) Run `mkdir -p data/{METR-LA,PEMS-BAY}` to create `/data/METR-LA` and `/data/PEMS-BAY` directories.
5) Follow the instructions from the [DCRNN repo](https://github.com/liyaguang/DCRNN) to download the `metr-la.h5` and `pems-bay.h5` datasets. Place the `metr-la.h5` file into the directory `/data/METR-LA/`. Place the `pems-bay.h5` file into the directory `/data/PEMS-BAY/`.
6) In the root project directory, process the `.h5` file with the following commands:

`python normalized_generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/METR-LA/metr-la.h5`

`python normalized_generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/PEMS-BAY/pems-bay.h5`

(Note: this script was adapted from [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet). The only modification was adding z-score normalization while creating the datasets.)

7) Open the `.ipynb` files and run the notebooks to evaluate the STBLIS-Net models and MLP/KDE baselines.