# VMNet: Voxel-Mesh Network for Geodesic-Aware 3D Semantic Segmentation

![Framework Fig](docs/network_detailed.jpg)

Created by Zeyu HU

## Introduction
This work is based on our paper 
[VMNet: Voxel-Mesh Network for Geodesic-Aware 3D Semantic Segmentation](https://arxiv.org/abs/2107.13824),
which appears at the IEEE International Conference on Computer Vision (ICCV) 2021. 

In recent years, sparse voxel-based methods have become the state-of-the-arts for 3D semantic segmentation of indoor scenes, thanks to the powerful 3D CNNs. Nevertheless, being oblivious to the underlying geometry, voxel-based methods suffer from ambiguous features on spatially close objects and struggle with handling complex and irregular geometries due to the lack of geodesic information. In view of this, we present Voxel-Mesh Network (VMNet), a novel 3D deep architecture that operates on the voxel and mesh representations leveraging both the Euclidean and geodesic information. Intuitively, the Euclidean information extracted from voxels can offer contextual cues representing interactions between nearby objects, while the geodesic information extracted from meshes can help separate objects that are spatially close but have disconnected surfaces. To incorporate such information from the two domains, we design an intra-domain attentive module for effective feature aggregation and an inter-domain attentive module for adaptive feature fusion. Experimental results validate the effectiveness of VMNet: specifically, on the challenging ScanNet dataset for large-scale segmentation of indoor scenes, it outperforms the state-of-the-art SparseConvNet and MinkowskiNet (74.6% vs 72.5% and 73.6% in mIoU) with a simpler network structure (17M vs 30M and 38M parameters).

## Citation
If you find our work useful in your research, please consider citing:

    @misc{hu2021vmnet,
          title={VMNet: Voxel-Mesh Network for Geodesic-Aware 3D Semantic Segmentation}, 
          author={Zeyu Hu and Xuyang Bai and Jiaxiang Shang and Runze Zhang and Jiayu Dong and Xin Wang and Guangyuan Sun and Hongbo Fu and Chiew-Lan Tai},
          year={2021},
          eprint={2107.13824},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

## Installation
* Our code is based on <a href="https://pytorch.org/">Pytorch</a>. Please make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. One configuration has been tested: 
     - Python 3.7
     - Pytorch 1.4.0
     - torchvision 0.5.0
     - CUDA 10.0
     - cudatoolkit 10.0.130
     - cuDNN 7.6.5

* VMNet depends on the <a href="https://github.com/rusty1s/pytorch_geometric">torch-geometric</a> and <a href="https://github.com/mit-han-lab/torchsparse">torchsparse</a> libraries. Please follow their installation instructions. One configuration has been tested, higher versions should work as well:
     - torch-geometric 1.6.3
     - torchsparse 1.1.0

* We adapted <a href="https://github.com/cnr-isti-vclab/vcglib">VCGlib</a> to generate pooling trace maps for vertex clustering and quadric error metrics.

      git clone https://github.com/cnr-isti-vclab/vcglib

      # QUADRIC ERROR METRICS
      cd vcglib/apps/tridecimator/
      qmake
      make

      # VERTEX CLUSTERING
      cd ../sample/trimesh_clustering
      qmake
      make

  Please add `vcglib/apps/tridecimator` and `vcglib/apps/sample/trimesh_clustering` to your environment path variable.

* Other dependencies. One configuration has been tested:
     - open3d 0.9.0
     - plyfile 0.7.3
     - scikit-learn 0.24.0
     - scipy 1.6.0
 
## Data Preparation
* Please refer to https://github.com/ScanNet/ScanNet and https://github.com/niessner/Matterport to get access to the ScanNet and Matterport dataset. Our method relies on the .ply as well as the .labels.ply files. We take ScanNet dataset as example for the following instructions.

* Create directories to store processed data.
     - 'path/to/processed_data/train/'
     - 'path/to/processed_data/val/'
     - 'path/to/processed_data/test/'

* Prepare train data.
      
      python prepare_data.py --considered_rooms_path dataset/data_split/scannetv2_train.txt --in_path path/to/ScanNet/scans --out_path path/to/processed_data/train/

* Prepare val data.
      
      python prepare_data.py --considered_rooms_path dataset/data_split/scannetv2_val.txt --in_path path/to/ScanNet/scans --out_path path/to/processed_data/val/

* Prepare test data.
      
      python prepare_data.py --test_split --considered_rooms_path dataset/data_split/scannetv2_test.txt --in_path path/to/ScanNet/scans_test --out_path path/to/processed_data/test/

## Train
* On train/val/test setting.
    
      CUDA_VISIBLE_DEVICES=0 python run.py --train --exp_name name_you_want --data_path path/to/processed_data
      
* On train+val/test setting (for ScanNet benchmark).

      CUDA_VISIBLE_DEVICES=0 python run.py --train_benchmark --exp_name name_you_want --data_path path/to/processed_data

## Inference
* Validation. <a href="https://drive.google.com/drive/folders/1G8ug8C7DCrHPeZ_91hM7TSVCTgbNajN0?usp=sharing">Pretrained model</a> (73.3% mIoU on ScanNet Val). Please download and put into directory check_points/val_split.

      CUDA_VISIBLE_DEVICES=0 python run.py --val --exp_name val_split --data_path path/to/processed_data

* Test. <a href="https://drive.google.com/drive/folders/1j5_uTL4xrsVZ5svPOp3LENcgvsJehDbx?usp=sharing">Pretrained model</a> (74.6% mIoU on ScanNet Test). Please download and put into directory check_points/test_split. TxT files for benchmark submission will be saved in directory test_results/.

      CUDA_VISIBLE_DEVICES=0 python run.py --test --exp_name test_split --data_path path/to/processed_data
      
## Acknowledgements
Our code is built upon <a href="https://github.com/rusty1s/pytorch_geometric">torch-geometric</a>, <a href="https://github.com/mit-han-lab/torchsparse">torchsparse</a> and <a href="https://github.com/VisualComputingInstitute/dcm-net">dcm-net</a>.

## License
Our code is released under MIT License (see LICENSE file for details).
