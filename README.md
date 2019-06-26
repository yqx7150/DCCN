# DCCN
The Code is created based on the method described in the following paper:  
A Comparative Study of CNN-based Super-resolution Methods in MRI Reconstruction and Its Beyond

# Our Previous ISBI Paper
A Comparative Study of CNN-based Super-resolution Methods in MRI Reconstruction

## Abstract
The progress of convolution neural network (CNN) based Super-resolution (SR) has shown its potential in image processing community. Meanwhile, Compressed Sensing MRI (CS-MRI) provides the possibility to accelerate the traditional acquisition process of MRI. In this work, on the basis of decomposing the cascade network to be a series of alternating CNN-based sub-network and data-consistency sub-network, we investigate the performance of the cascade networks in CS-MRI by employing various CNN-based super-resolution methods in the CNN-based sub-network. Furthermore, realizing that existing methods only explore dense connection in the CNN-based sub-network which insufficiently explore the feature information, we propose a dense connected cascade network (DCCN) for more accurate MR reconstruction. Specifically, DCCN network densely connects both CNN-based sub-network and data-consistency sub-network, thus takes advantage of the data-consistency of k-space data in a densely connected fashion. Experimental results on various MR data demonstrated that DCCN is superior to current cascade networks in reconstruction quality.

### The flowchart of DCCN. 
![repeat-DCCN](https://github.com/yqx7150/DCCN/blob/master/flow.png)

## Requirements and Dependencies
    theano
    cuda
    cudnn
    python
    
## cmd
'./DCCN.py' is the demo of DCCN.

### The output images of DCCN. 
![repeat-DCCN](https://github.com/yqx7150/DCCN/blob/master/models/d5_c5/radial85/epoch1458_im0.png)
![repeat-DCCN](https://github.com/yqx7150/DCCN/blob/master/models/d5_c5/radial85/epoch1458_im1.png)


Previous ISBI Paper
    @article{zeng2019isbi,   
    title=A Comparative Study of CNN-based Super-resolution Methods in MRI Reconstruction},   
    author={Wei Zeng, Jie Peng, Shanshan Wang, Zhicheng Li, Qiegen Liu, Dong Liang},   
    conference={IEEE 16th International Symposium on Biomedical Imaging},   
    year={2019},   
    }
