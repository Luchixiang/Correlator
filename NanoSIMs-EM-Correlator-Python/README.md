# NanoSIMS EM Correlator Python version
This repository contains the Python source code for our NanoSIMS EM Correlator, which allows automatic registration between EM image and the NanoSIMS image

![image-20240108200232561](./Figure1_v2.png)


## Installation
```Shell
git clone https://github.com/Luchixiang/NanoSIMS_EM_Correlator
cd NanoSIMS_Stabilizer_Correlator/NanoSIMs-EM-Correlator-Python
```
## Requirements
The code has been tested with PyTorch 2.0. 
```Shell
conda create --name stabilizer python=3.9
conda activate stabilizer
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch 
pip install -r requirments.txt
```

## Register EM and NanoSIMS image
We currently only support .nrrd file. If you only have .im file, you can batch convert the .im to .nrrd with OpenMIMS plugin. 
```Shell
cd core
python nanosims_em_correlate.py --em_path image_path --nano_path nano_path --em_res 6 --nano_res 97.6 --channel 32S
```
Please replace the ``--em_path`` with the EM file and ``--nano_path`` with the NanoSIMS file you want to register. 

Also please indicate the signal channel that is used to calculate the transformation map and apply it to other channels. Strong signal channels are recommended such as  32S. Also, please indicate your EM pixel size and NanoSIMS pixel size. 

We provide a demo file. To register the demo file:
```shell
python nanosims_em_correlate.py --em_path ../demo-frames/liver_em_image.png --nano_path ../demo-frames/liver.nrrd --em_res 6 --nano_res 97.6 --channel 32S
```
A ``_aligned.nrrd`` will be saved in the same folder as the input NanoSIMS file. 

