# DCCN
The Code is created based on the method described in the following paper:  
A Comparative Study of CNN-based Super-resolution Methods in MRI Reconstruction and Its Beyond

# Our Previous ISBI Paper
A Comparative Study of CNN-based Super-resolution Methods in MRI Reconstruction

## Abstract
The progress of convolution neural network (CNN) based Super-resolution (SR) has shown its potential in image processing community. Meanwhile, Compressed Sensing MRI (CS-MRI) provides the possibility to accelerate the traditional acquisition process of MRI. In this work, on the basis of decomposing the cascade network to be a series of alternating CNN-based sub-network and data-consistency sub-network, we investigate the performance of the cascade networks in CS-MRI by employing various CNN-based super-resolution methods in the CNN-based sub-network. Furthermore, realizing that existing methods only explore dense connection in the CNN-based sub-network which insufficiently explore the feature information, we propose a dense connected cascade network (DCCN) for more accurate MR reconstruction. Specifically, DCCN network densely connects both CNN-based sub-network and data-consistency sub-network, thus takes advantage of the data-consistency of k-space data in a densely connected fashion. Experimental results on various MR data demonstrated that DCCN is superior to current cascade networks in reconstruction quality.

### Overall structure of the DCCN. 
<div align=center><img width="600" height="200" src="https://github.com/yqx7150/DCCN/blob/master/flow.png"/></div>
Overall structure of the DCCN. It is composed of five identical basic block cascades; each basic block consists of a CNN (one RDB) and a data consistency layer (DC).

### The CNN structure of RDN block in DCCN
<div align=center><img width="600" height="130" src="https://github.com/yqx7150/DCCN/blob/master/flow1.png"/></div>
The CNN structure of RDN block in DCCN. The Convolution layers and ReLU layers are denoted as “C” and “R”, respectively. The “concat” means all the input data of this layer will be concatenated in the first dimension. 

### Structure of “Basic CNN Unit”.
<div align=center><img width="600" height="95" src="https://github.com/yqx7150/DCCN/blob/master/flow2.png"/></div>


# Reconstruction Results of Different Methods. 
![repeat-DCCN](https://github.com/yqx7150/DCCN/blob/master/2.png)  
Reconstruction results by various methods at 90% Pseudo radial undersampling. Red boxes illustrate the enlarged view. From left to right: Fully-sampled MRI image, udersampling MRI image, NLR-CS, CN-CNN, CN-DenseNet, CN-EDSR, CN-RDN and DCCN.

![repeat-DCCN](https://github.com/yqx7150/DCCN/blob/master/3.png)  
Reconstruction results by various methods at 85% 2D random undersampling. Red boxes illustrate the enlarged view. From left to right: Fully-sampled MRI image, udersampling MRI image, NLR-CS, CN-CNN, CN-DenseNet, CN-EDSR, CN-RDN and DCCN.

![repeat-DCCN](https://github.com/yqx7150/DCCN/blob/master/4.png)  
Reconstruction results by various methods at 85% 1D Cartesian undersampling. Red boxes illustrate the enlarged view. From left to right: Fully-sampled MRI image, udersampling MRI image, NLR-CS, CN-CNN, CN-DenseNet, CN-EDSR, CN-RDN and DCCN.

## Requirements and Dependencies
    theano
    cuda
    cudnn
    python3
    
## cmd
'python DCCN.py' is the demo of DCCN.

# Previous ISBI Paper
    @article{zeng2019isbi,   
    title=A Comparative Study of CNN-based Super-resolution Methods in MRI Reconstruction},   
    author={Wei Zeng, Jie Peng, Shanshan Wang, Zhicheng Li, Qiegen Liu, Dong Liang},   
    conference={IEEE 16th International Symposium on Biomedical Imaging},   
    year={2019},   
    }


## Other Related Projects
  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)
  
  * IFR-Net: Iterative Feature Refinement Network for Compressed Sensing MRI [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/8918016)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/IFR-Net-Code)
    
  * Iterative scheme-inspired network for impulse noise removal [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10044-018-0762-8)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/IIN-Code)

