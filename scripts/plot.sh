# #Dataset 1
# python plot_caloscore.py  --config config_dataset1.json --sample --nsplit 5
# python plot_caloscore.py  --config config_dataset1.json --sample --nsplit 5 --distill --factor 64
# python plot_caloscore.py  --config config_dataset1.json --sample --nsplit 5 --distill --factor 512
# #Dataset 2
# python plot_caloscore.py  --config config_dataset2.json --sample --nsplit 100
# python plot_caloscore.py  --config config_dataset2.json --sample --nsplit 100 --distill --factor 64
# python plot_caloscore.py  --config config_dataset2.json --sample --nsplit 100 --distill --factor 512
# #Dataset 3
# python plot_caloscore.py --distill --factor 4 --config config_dataset3.json --sample --nsplit 150
# python plot_caloscore.py --distill --factor 64 --config config_dataset3.json --sample --nsplit 150
# python plot_caloscore.py --distill --factor 512 --config config_dataset3.json --sample --nsplit 150
#Dataset 1
python plot_caloscore.py  --config config_dataset1.json 
python plot_caloscore.py  --config config_dataset1.json  --factor 64
python plot_caloscore.py  --config config_dataset1.json  --factor 512
python plot_caloscore.py  --config config_dataset1.json  --model all
#Dataset 2
python plot_caloscore.py  --config config_dataset2.json 
python plot_caloscore.py  --config config_dataset2.json  --factor 64
python plot_caloscore.py  --config config_dataset2.json  --factor 512
python plot_caloscore.py  --config config_dataset2.json  --model all
#Dataset 3

python plot_caloscore.py  --config config_dataset3.json --factor 64 
python plot_caloscore.py  --config config_dataset3.json --factor 512

#
# python evaluate.py -i /global/cfs/cdirs/m3929/SCRATCH/FCC/generated_dataset1_CaloScore_1.h5 -r /global/cfs/cdirs/m3929/SCRATCH/FCC/dataset_1_photons_2.hdf5 -m cls-low -d 1-photons
# python evaluate.py -i /global/cfs/cdirs/m3929/SCRATCH/FCC/generated_dataset2_CaloScore_1.h5 -r /global/cfs/cdirs/m3929/SCRATCH/FCC/dataset_2_2.hdf5 -m cls-low -d 2
# python evaluate.py -i /global/cfs/cdirs/m3929/SCRATCH/FCC/generated_dataset3_CaloScore_1.h5 -r /global/cfs/cdirs/m3929/SCRATCH/FCC/dataset_3_3.hdf5 -m cls-low -d 3
