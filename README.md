[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

# milliEgo
### [Youtube](https://www.youtube.com/watch?v=I9vjoKGY2ts&feature=youtu.be) | [Paper](https://arxiv.org/abs/2006.02266) <br>

Simplified [docker](https://www.docker.com/) version for the implementation of our 6-DOF Egomotion Estimation method via a single-chip mmWave radar ([TI AWR1843](https://www.ti.com/product/AWR1843)) and a commercial-grade IMU. Our method is the first-of-its-kind DNN based odometry approach that can estimate the egomotion from the sparse and noisy data returned by a single-chip mmWave radar. <br><br>
[milliEgo: Single-chip mmWave Aided Egomotion Estimation with Deep Sensor Fusion](https://arxiv.org/abs/2006.02266)  
Chris Xiaoxuan Lu, Muhamad Risqi U. Saputra, Peijun Zhao, Yasin Almalioglu, Pedro P. B. de Gusmao, Changhao Chen, Ke Sun, Niki Trigoni, Andrew Markham
In [SenSys 2020](https://www.sigmobile.org/sensys/2020/).  

## Prerequisites

- TensorFlow 2.X

### Pre-trained mmWave Radar Feature Extractor and milliEgo model
- After git clone this repository, enter the project directory,
```
mkdir -p models/cross-mio
```
- Download the trained milliEgo model '18'` from [here](https://drive.google.com/file/d/1KxUUat5yP1oAsUSg0T6n3JZh94sKWHy2/view?usp=sharing).
-  Unzip and put it in `./models/cross-mio/`.

### Dataset
- To train and test a model, please download our dataset from [here](https://www.dropbox.com/s/q6z81pe1mxr0iyo/milliVO_dataset.zip?dl=0) (dropbox link).
- Specify the path of the dataset in "multimodal_data_dir" in config.yaml
- Run dataset_convert.py to convert the training set to tf.data.Dataset, it will be store in the path specified in "tf_data_dir" in config.yaml

### Testing
```
python test.py
```

### Training
```
python train.py
```



## Citation

If you find this useful for your research, please use the following.

```
@inproceedings{lu2020milliego,
  title={milliEgo: single-chip mmWave radar aided egomotion estimation via deep sensor fusion},
  author={Lu, Chris Xiaoxuan and Saputra, Muhamad Risqi U and Zhao, Peijun and Almalioglu, Yasin and de Gusmao, Pedro PB and Chen, Changhao and Sun, Ke and Trigoni, Niki and Markham, Andrew},
  booktitle={Proceedings of the 18th Conference on Embedded Networked Sensor Systems (SenSys)},
  year={2020}
}
```
