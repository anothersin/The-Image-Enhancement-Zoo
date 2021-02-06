# The Image Enhancement Zoo
By Yansheng Qiu, Kui Jiang

## Description
* Image Enhancement Zoo: A list of [deraining](#Deraining), [dehazing](#Dehazing) and [brightening](#Brightening) methods. Papers, codes and datasets are maintained.
* Thanks for the sharing of [Resources for Low Light Image Enhancement](https://github.com/dawnlh/low-light-image-enhancement-resources), [low-light-image-enhancment](https://github.com/Mengke-Yuan/low-light-image-enhancment), [DehazeZoo](https://github.com/cxtalk/DehazeZoo).

## 2 Image Quality Metrics
* PSNR (Peak Signal-to-Noise Ratio) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4550695) [[matlab code]](https://www.mathworks.com/help/images/ref/psnr.html) [[python code]](https://github.com/aizvorski/video-quality)
* SSIM (Structural Similarity) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1284395) [[matlab code]](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) [[python code]](https://github.com/aizvorski/video-quality/blob/master/ssim.py)
* VIF (Visual Quality) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1576816) [[matlab code]](http://sse.tongji.edu.cn/linzhang/IQA/Evalution_VIF/eva-VIF.htm)
* FSIM (Feature Similarity) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5705575) [[matlab code]](http://sse.tongji.edu.cn/linzhang/IQA/FSIM/FSIM.htm)
* NIQE (Naturalness Image Quality Evaluator) [[paper]](http://live.ece.utexas.edu/research/Quality/niqe_spl.pdf) [[matlab code]](http://live.ece.utexas.edu/research/Quality/index_algorithms.htm) [[python code]](https://github.com/aizvorski/video-quality/blob/master/niqe.py)
## Brightening

### Datasets
- VV, LIME, NPE-series, DICM, MEF
  - only low-light images without corresponding high-light ground truth
  - [Download](https://drive.google.com/drive/folders/0B_FjaR958nw_djVQanJqeEhUM1k?usp=sharing)
  - Thanks to [baidut](https://github.com/baidut/BIMEF) for collection

- ExDARK
  - only low-light image without corresponding high-light ground truth
  - [[paper]](http://cs-chan.com/doc/cviu.pdf) [[Homepage]](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset)

- LOL
  - old: [[papper]](https://github.com/daooshee/BMVC2018website/blob/master/chen_bmvc18.pdf) [[Homepage]](https://daooshee.github.io/BMVC2018website/)
  - new: [[dataset baidu:r0xd]](https://pan.baidu.com/s/1-j6_3G9WHS8rkEBzHN7QvQ)

- SID [[paper]](https://arxiv.org/abs/1805.01934) [[Homepage]](http://cchen156.web.engr.illinois.edu/SID.html)

- MIT-Adobe FiveK [[paper]](http://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf) [[Homepage]](https://data.csail.mit.edu/graphics/fivek/)

- SICE [[paper]](https://ieeexplore.ieee.org/document/8259342/citations?tabFilter=papers) [[Homepage]](https://github.com/csjcai/SICE)
  
- ELD [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper.pdf) [[Homepage]](https://github.com/Vandermode/ELD)

- VIP-LowLight [[Homepage]](https://uwaterloo.ca/vision-image-processing-lab/research-demos/vip-lowlight-dataset)

- ReNOIR [[paper]](https://arxiv.org/abs/1409.8230) [[Homepage]](http://adrianbarburesearch.blogspot.com/p/renoir-dataset.html)

- DPED [[paper]](https://arxiv.org/pdf/1704.02470.pdf) [[Homepage]](http://people.ee.ethz.ch/~ihnatova/)
  
### Papers
### 2020

- **[RLI-DaE]** Learning to Restore Low-Light Images via Decomposition-and-Enhancement (**CVPR 2020**) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Learning_to_Restore_Low-Light_Images_via_Decomposition-and-Enhancement_CVPR_2020_paper.pdf)]

- **[Pb-NFM]** A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising (**CVPR 2020**, paper for [ELD dataset](#ELD)) [[paper](https://arxiv.org/abs/2003.12751)][[code](https://github.com/Vandermode/ELD)]

- **[DALE]** DALE : Dark Region-Aware Low-light Image Enhancement (**BMVC 2020**) [[paper](https://www.bmvc2020-conference.com/assets/papers/1025.pdf) [[code]](https://github.com/dokyeongK/DALE)

- **[LLPackNet]** Towards Fast and Light-Weight Restoration of Dark Images (**BMVC 2020**) [[paper](https://www.bmvc2020-conference.com/assets/papers/0145.pdf)][[code](https://github.com/MohitLamba94/LLPackNet)]

- **[Zero-DCE]** Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (**CVPR 2020**) [[paper](https://arxiv.org/abs/2001.06826v2)][[homepage](https://li-chongyi.github.io/Proj_Zero-DCE.html)][[code](https://github.com/Li-Chongyi/Zero-DCE)]

- **[DRBN]** From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (**CVPR 2020**) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_From_Fidelity_to_Perceptual_Quality_A_Semi-Supervised_Approach_for_Low-Light_CVPR_2020_paper.pdf)][[code]](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)

- **[LR3M]** LR3M: Robust Low-Light Enhancement via Low-Rank Regularized Retinex Model (**TIP 2020**) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9056796)

- **[DRBN]** From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (**CVPR 2020**) [[paper]](https://ieeexplore.ieee.org/document/9156559) [[code]](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)

- **[STARnet]** Space-Time-Aware Multi-Resolution Video Enhancement (**CVPR 2020**) [[homepage]](https://alterzero.github.io/projects/STAR.html) [[code]](https://github.com/alterzero/STARnet) [[paper]](https://alterzero.github.io/projects/star_cvpr2020.pdf)

- **[DeepLPF]** DeepLPF: Deep Local Parametric Filters for Image Enhancement (**CVPR 2020**) [[code]](https://github.com/sjmoran/DeepLPF)  [[paper]](https://arxiv.org/abs/2003.13985)

- **[STAR]** STAR: A Structure and Texture Aware Retinex Model (**TIP2020**) [[code]](https://github.com/csjunxu/STAR-TIP2020) [[paper]](https://ieeexplore.ieee.org/document/9032356)

- **[MIRNet]** Learning Enriched Features for Real Image Restoration and Enhancement (**ECCV 2020**) [[paper]](https://arxiv.org/abs/2003.06792) [[code]](https://github.com/swz30/MIRNet)

- **[CSRnet]** Conditional Sequential Modulation for Efficient Global Image Retouching (**ECCV 2020**) [[paper]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580664.pdf)  [[code]](https://github.com/hejingwenhejingwen/CSRNet)

- **[Flickr]** Unpaired Image Enhancement with Quality-Attention Generative Adversarial Network (**ACM MM 2020**) [[paper]](https://dl.acm.org/doi/10.1145/3394171.3413839)

- **[RT-VENet]** RT-VENet: A Convolutional Network for Real-time Video Enhancement (**ACM MM 2020**) [[paper]](https://dl.acm.org/doi/pdf/10.1145/3394171.3413951)

- **Integrating Semantic Segmentation and Retinex Model for Low Light Image Enhancement**  (**ACM MM 2020**) [[paper]](http://39.96.165.147/Pub%20Files/2020/fmh_mm20.pdf) [[homepage]](https://mm20-semanticreti.github.io/) [dataset [google](https://drive.google.com/drive/folders/1rdMI2oUBv8eYrfvnmQ2ihLjzLh831Sff?usp=sharing) or [baidu:2wml](https://pan.baidu.com/s/1zx2tG9trnSrowz0wHM8phg)]

- **Low-Light Image Enhancement with Semi-Decoupled Decomposition** (**TMM 2020**) [[paper]](https://ieeexplore.ieee.org/document/8970535)  [[code]](https://github.com/hanxuhfut/Code)

- **Learning to Restore Low-Light Images via Decomposition-and-Enhancement** (**CVPR 2020**) [[paper]](https://ieeexplore.ieee.org/document/9156446)


### 2019

- **[EnlightenGAN]** EnlightenGAN: Deep Light Enhancement without Paired Supervision [[paper](https://arxiv.org/pdf/1906.06972)][[code](https://github.com/TAMU-VITA/EnlightenGAN)]

- **[DCGANs]** Deep Learning for Robust end-to-end Tone Mapping (**BMVC 2019**) [[paper](https://bmvc2019.org/wp-content/uploads/papers/0849-paper.pdf)]

- **[RJI]** Robust Joint Image Reconstruction from Color and Monochrome Cameras (**BMVC 2019**) [[paper](https://bmvc2019.org/wp-content/uploads/papers/0754-paper.pdf)]

- **[KinD]** Kindling the Darkness: A Practical Low-light Image Enhancer (**ACM MM 2019**) [[paper]](https://arxiv.org/pdf/1905.04161)  [[code](https://github.com/zhangyhuaee/KinD)]

- **[DeepUPE]** Underexposed Photo Enhancement using Deep Illumination Estimation (**CVPR 2019**) [[paper](https://drive.google.com/file/d/1CCd0NVEy0yM2ulcrx44B1bRPDmyrgNYH/view)][[code (only test)](https://github.com/wangruixing/DeepUPE)]

- **[EEMEFN]** EEMEFN: Low-Light Image Enhancement via Edge-Enhanced Multi-Exposure Fusion Network (**AAAI 2019**) [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/7013/6867)   [[code]](https://github.com/MinfengZhu/EEMEFN)

- **[ALSM]** Low-Light Image Enhancement via the Absorption Light Scattering Model (**TIP 2019**) [[paper](https://doi.org/10.1109/TIP.2019.2922106) [[code]](https://github.com/blisswyf/ALSMdemo)

- **Low-Light Image Enhancement via a Deep Hybrid Network** (**TIP 2019**) [[paper]](https://ieeexplore.ieee.org/document/8692732) 

- **Enhancing Low Light Videos by Exploring High Sensitivity Camera Noise** (**ICCV 2019**) [[paper]](https://ieeexplore.ieee.org/document/9011000)



### 2018

- **[MBLLEN]** MBLLEN: Low-light Image/Video Enhancement Using CNNs (**BMVC 2018**) [[paper](http://bmvc2018.org/contents/papers/0700.pdf)][[code](https://github.com/Lvfeifan/MBLLEN)]

- **[GLADNet]** GLADNet: Low-Light Enhancement Network with Global Awareness (**FG 2018**) [[paper](https://ieeexplore.ieee.org/document/8373911)][[code](https://github.com/weichen582/GLADNet)]

- **[Retinex-Net]** Deep Retinex Decomposition for Low-Light Enhancement (**BMVC 2018**) [[paper](https://arxiv.org/pdf/1808.04560)][[code](https://github.com/weichen582/RetinexNet)][[code-pytorch](https://github.com/FunkyKoki/RetinexNet_PyTorch)]

- **[SID]** Learning to See in the Dark (**CVPR 2018**) [[paper](http://cchen156.web.engr.illinois.edu/paper/18CVPR_SID.pdf)][[code](https://github.com/cchen156/Learning-to-See-in-the-Dark)][[code-pytorch](https://github.com/cydonia999/Learning_to_See_in_the_Dark_PyTorch)]

- **[MBLLEN]** MBLLEN: Low-light Image/Video Enhancement Using CNNs (**BMVC 2018**) [[[homepage]](http://phi-ai.org/project/MBLLEN/default.htm) [[code]](https://github.com/Lvfeifan/MBLLEN) [[paper]](http://bmvc2018.org/contents/papers/0700.pdf)

- **[SICE]** Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images (**TIP 2018**) [[code]](https://github.com/csjcai/SICE) [[paper]](https://doi.org/10.1109/TIP.2018.2794218)

- **[White-Box]** Exposure: A White-Box Photo Post-Processing Framework (**TOG 2018**) [[code]](https://github.com/yuanming-hu/exposure) [[paper]](https://doi.org/10.1145/3181974)

- **Structure-Revealing Low-Light Image Enhancement Via Robust Retinex Model** (**TIP 2018**) [[code]](https://github.com/martinli0822/Low-light-image-enhancement)  [[paper]](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/2018-TIP-Structure-Revealing-Low-Light-Image-Enhancement-Via-Robust-Retinex-Model.pdf)



### before

- **[HDRNet]** Deep Bilateral Learning for Real-Time Image Enhancement (**SIGGRAPH 2017**) [[paper](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf)][[code](https://github.com/google/hdrnet)]

- **[LLCNN]**  LLCNN: A convolutional neural network for low-light image enhancement (**VCIP 2017**) [[paper](https://ieeexplore.ieee.org/abstract/document/8305143)][[code](https://github.com/BestJuly/LLCNN)]

- **[LIME]** LIME: Low-Light Image Enhancement via Illumination Map Estimation (**TIP 2017**)  and LIME: A Method for Low-light IMage Enhancement (**ACM MM 2016**)[[homepage]](https://sites.google.com/view/xjguo/lime) [[Code_official]](https://drive.google.com/file/d/0BwVzAzXoqrSXb3prWUV1YzBjZzg/view) [[Code1]](https://github.com/Sy-Zhang/LIME) [[Code2]](https://github.com/estija/LIME) [[Code3]](https://github.com/pvnieo/Low-light-Image-Enhancement) [[paper]](https://ieeexplore.ieee.org/document/7782813) 

- **[JieP]**  A Joint Intrinsic-Extrinsic Prior Model for Retinex (**ICCV 2017**) [[homepage]](http://caibolun.github.io/JieP/) [[code]](https://github.com/caibolun/JieP/) [[paper]](http://caibolun.github.io/papers/JieP.pdf)

- **A New Low-Light Image Enhancement Algorithm Using Camera Response Model** (**ICCVW 2017**) [[Code]](https://github.com/baidut/OpenCE/blob/master/ours/Ying_2017_ICCV.m) [[Pdf]](http://ieeexplore.ieee.org/document/8265567/) 

- **DSLR Quality Photos on Mobile Devices with Deep Convolutional Networks**  (**ICCV 2017**) [[paper]](https://arxiv.org/abs/1704.02470)

- **[SRIE]** A Probabilistic Method for Image Enhancement With Simultaneous Illumination and Reflectance Estimation (**TIP 2015**) and A Weighted Variational Model for Simultaneous Reflectance and Illumination Estimation(**CVPR 2016**)[[Pdf_1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7229296) [[Pdf_2]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Fu_A_Weighted_Variational_CVPR_2016_paper.pdf) [[Code_1]](codes/PM_SIRE.zip) [[Code_2]](codes/WV_SIRE.zip)


## Dehazing

### Datasets

- 3R [[paper](https://arxiv.org/abs/2008.03864)][[dataset](https://github.com/chaimi2013/3R)]

- KITTI [paper][[dataset](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php)]

- RESIDE [[paper](https://arxiv.org/pdf/1712.04143.pdf)][[dataset](https://sites.google.com/view/reside-dehaze-datasets)]

- HazeRD [[paper](http://www.ece.rochester.edu/~gsharma/papers/Zhang_ICIP2017_HazeRD.pdf)][[dataset](https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/)]

- SceneNet [paper][[dataset](https://robotvault.bitbucket.io/scenenet-rgbd.html)]

- I-HAZE [[paper](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/O-HAZE.pdf)][[dataset](http://www.vision.ee.ethz.ch/ntire18/i-haze/)]

- O-HAZE [[paper](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/O-HAZE.pdf)][[dataset](http://www.vision.ee.ethz.ch/ntire18/o-haze/)]

- D-HAZY [[paper](http://www.meo.etc.upt.ro/AncutiProjectPages/D_Hazzy_ICIP2016/D_HAZY_ICIP2016.pdf)][[dataset](https://www.researchgate.net/publication/307516141_D-HAZY_A_dataset_to_evaluate_quantitatively_dehazing_algorithms)]

- Middlebury [[paper](http://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf)][[dataset](http://vision.middlebury.edu/stereo/data/scenes2014/)]

- 3DRealisticScene [[paper](https://arxiv.org/abs/2004.08554)][[dataset](https://github.com/liruoteng/3DRealisticSceneDehaze)]

- NYU Depth Dataset V2 [[paper](https://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf)][[dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)]

### Papers
### 2020

- **[PFDN]** Physics-based Feature Dehazing Networks (**ECCV 2020**) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750188.pdf)]

- **[HardGAN]** HardGAN: A Haze-Aware Representation Distillation GAN for Single Image Dehazing (**ECCV 2020**) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510715.pdf)][[code](https://github.com/huangzilingcv/HardGAN)]

- **[Dehaze-GLCGAN]** Dehaze-GLCGAN: Unpaired Single Image De-hazing via Adversarial Training [[paper](http://xxx.itp.ac.cn/abs/2008.06632)]

- **[ND-Net]** Nighttime Dehazing with a Synthetic Benchmark (**ACM MM 2020**)[[paper](https://arxiv.org/abs/2008.03864)][[code](https://github.com/chaimi2013/3R)]

- **[MI-Net]** Implicit Euler ODE Networks for Single-Image Dehazing (**CVPRW 2020**) [[paper](https://arxiv.org/abs/2007.06443)]

- **[BidNet]** BidNet: Binocular Image Dehazing without Explicit Disparity Estimation (**CVPR 2020**) [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_BidNet_Binocular_Image_Dehazing_Without_Explicit_Disparity_Estimation_CVPR_2020_paper.pdf)]

- **[MSBDN]** Multi-Scale Boosted Dehazing Network with Dense Feature Fusion (**CVPR 2020**) [[paper](https://arxiv.org/abs/2004.13388)][[code](https://github.com/BookerDeWitt/MSBDN-DFF)]

- **[DA_dahazing]** Domain Adaptation for Image Dehazing (**CVPR 2020**) [[paper](https://arxiv.org/abs/2005.04668)][[code](https://github.com/HUSTSYJ/DA_dahazing)][[homepage](https://sites.google.com/site/renwenqi888)]

- **[FD-GAN]** FD-GAN: Generative Adversarial Networks with Fusion-discriminator for Single Image Dehazing (**AAAI 2020**) [[paper](https://arxiv.org/abs/2001.06968)][[code](https://github.com/WeilanAnnn/FD-GAN)]

- **[FFA-Net]** FFA-Net: Feature Fusion Attention Network for Single Image Dehazing (**AAAI 2020**) [[paper](https://arxiv.org/abs/1911.07559)][[code](https://github.com/zhilin007/FFA-Net)]

- **Accurate Transmission Estimation for Removing Haze and Noise from a Single Image**  (**TIP 2020**) [[paper](https://ieeexplore.ieee.org/document/8891906)]

- **Single Image Dehazing via Multi-Scale Convolutional Neural Networks with Holistic Edges**  (**IJCV 2020**) [[paper](https://link.springer.com/article/10.1007%2Fs11263-019-01235-8)]

- **Fast Deep Multi-patch Hierarchical Network for Nonhomogeneous Image Dehazing**  (**CVPRW 2020**) [[paper](https://arxiv.org/abs/2005.05999)]

- **Transmission Map and Atmospheric Light Guided Iterative Updater Network for Single Image Dehazing**  (**CVPR 2020**) [[paper](http://xxx.itp.ac.cn/abs/2008.01701)][[code](https://github.com/aupendu/iterative-dehaze)]

- **Color Cast Dependent Image Dehazing via Adaptive Airlight Refinement and Non-linear Color Balancing** (**TSVT 2020**)[[paper](https://ieeexplore.ieee.org/document/9134933)][[code](https://github.com/m14roy/CC_AA_NCB_Img_Dehaze/tree/master/)]

### 2019

- **[GridDehazeNet]** GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing (**ICCV 2019**) [[paper](https://arxiv.org/abs/1908.03245)][[code](https://github.com/proteus1991/GridDehazeNet)]

- **[FastNet]** Feature Forwarding for Efficient Single Image Dehazing (**CVPRW 2019**)[[paper](https://arxiv.org/abs/1904.09059)]

- **[Cycle-Defog2Refog]** End-to-End Single Image Fog Removal using Enhanced Cycle Consistent Adversarial Networks (**TIP 2019**) [[paper](https://arxiv.org/abs/1902.01374)]

- **[VDHNet]** Deep Video Dehazing with Semantic Segmentation (**TIP 2019**) [[paper](https://ieeexplore.ieee.org/document/8492451)]

- **Learning Interleaved Cascade of Shrinkage Fields for Joint Image Dehazing and Denoising**  (**TIP 2019**) [[paper](https://ieeexplore.ieee.org/document/8852852)]

- **Semi-Supervised Image Dehazing**  (**TIP 2019**) [[paper](https://ieeexplore.ieee.org/abstract/document/8902220/)]

- **Benchmarking Single Image Dehazing and Beyond**  (**TIP 2019**) [[paper](https://arxiv.org/abs/1712.04143)][[homepage](https://sites.google.com/site/boyilics/website-builder/reside)]


### 2018

- **[GFN]** Gated Fusion Network for Single Image Dehazing (**CVPR 2018**) [[paper](https://arxiv.org/abs/1804.00213)][[code](https://github.com/rwenqi/GFN-dehazing)][[homepage](https://sites.google.com/site/renwenqi888/research/dehazing/gfn)]

- **[FEED-Net]** FEED-Net: Fully End-To-End Dehazin (**ICME 2018**) [[paper]](https://www.researchgate.net/publication/328245750_Feed-Net_Fully_End-to-End_Dehazing)

- **[DCPDN]** Densely Connected Pyramid Dehazing Network (**CVPR 2018**) [[paper](https://arxiv.org/abs/1803.08396)][[code](https://github.com/hezhangsprinter/DCPDN)]

- **[Cycle-Dehaze]** Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing (**CVPRW 2018**) [[paper](https://arxiv.org/abs/1805.05308v1)]

- **Towards Perceptual Image Dehazing by Physics-based Disentanglement and Adversarial Training**  (**AAAI 2018**) [[paper]](http://legacydirs.umiacs.umd.edu/~xyang35/files/perceptual-image-dehazing.pdf)

### Before

- **[AOD-Net]** An All-in-One Network for Dehazing and Beyond (**ICCV 2017**) [[paper](https://arxiv.org/pdf/1707.06543.pdf)][[code](https://github.com/MayankSingal/PyTorch-Image-Dehazing)][[homepage](https://sites.google.com/site/boyilics/website-builder/project-page)]

- **[DehazeNet]** DehazeNet: An end-to-end system for single image haze removal (**TIP 2016**) [[paper](http://caibolun.github.io/papers/DehazeNet.pdf)][[code](https://github.com/caibolun/DehazeNet)][[homepage](http://caibolun.github.io/DehazeNet/)]

- **A fast single image haze removal algorithm using color attenuation prior**  (**TIP 2015**) [[paper](https://ieeexplore.ieee.org/document/7128396)]

- **Single Image Dehazing via Multi-Scale Convolutional Neural Networks** (**ECCV 2016**) [[paper](https://drive.google.com/open?id=0B7PPbXPJRQp3TUJ0VjFaU1pIa28)][[code](https://sites.google.com/site/renwenqi888/research/dehazing/mscnndehazing/MSCNN_dehazing.zip?attredirects=0&d=1)][[homepage](https://sites.google.com/site/renwenqi888/research/dehazing/mscnndehazing)]

- **Single Image Haze Removal Using Dark Channel Prior** (**CVPR 2009**) [[paper](http://www.jiansun.org/papers/Dehaze_CVPR2009.pdf)]


## Deraining

### Datasets

#### Synthetic Datasets
- Rain12 [[paper](https://ieeexplore.ieee.org/document/7780668/)][[dataset](http://yu-li.github.io/paper/li_cvpr16_rain.zip)]
- Rain100L_old_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
  - Rain100L_new_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
- Rain100H_old_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md)]
  - Rain100H_new_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
- Rain800 [[paper](https://arxiv.org/abs/1701.05957)][[dataset](https://github.com/hezhangsprinter/ID-CGAN)]
- Rain1200 [[paper](https://arxiv.org/abs/1802.07412)][[dataset](https://github.com/hezhangsprinter/DID-MDN)]
- Rain1400 [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)][[dataset](https://xueyangfu.github.io/projects/cvpr2017.html)]
- Heavy Rain Dataset [[paper](http://export.arxiv.org/pdf/1904.05050)][[dataset](https://drive.google.com/file/d/1rFpW_coyxidYLK8vrcfViJLDd-BcSn4B/view)]

#### Real-world Datasets
- Practical_by_Yang [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
- Practica_by_Zhang [[paper](https://arxiv.org/abs/1701.05957)][[dataset](https://github.com/hezhangsprinter/ID-CGAN)]
- Real-world Paired Rain Dataset [[paper](https://arxiv.org/pdf/1904.01538.pdf)][[dataset]](https://stevewongv.github.io/derain-project.html)

### Papers

### 2021
**[WDNet]**
- **[DualGCN]** Rain Streak Removal via Dual Graph Convolutional Network (**AAAI 2021**) [[paper](https://xueyangfu.github.io/paper/2021/AAAI/Preprint.pdf)][[code](https://xueyangfu.github.io/paper/2021/AAAI/code.zip)][[homepage](https://xueyangfu.github.io)]

### 2020

- **[WDNet]** Wavelet-Based Dual-Branch Network for Image Demoir´eing (**ECCV 2020**) [[paper](https://arxiv.org/pdf/2008.00823.pdf)]

- **[Rethinking Image Deraining]** Rethinking Image Deraining via Rain Streaks and Vapors (**ECCV 2020**) [[paper](https://arxiv.org/pdf/2008.00823.pdf)][[code](https://github.com/yluestc/derain)][[homepage](https://github.com/yluestc)]

- **[JDNet]** Joint Self-Attention and Scale-Aggregation for Self-Calibrated Deraining Network (**ACM MM 2020**) [[paper](https://arxiv.org/pdf/2008.02763.pdf)][[code](https://github.com/Ohraincu/JDNet)][[homepage](https://github.com/Ohraincu)]

- **[DCSFN]** DCSFN: Deep Cross-scale Fusion Network for Single Image Rain Removal (**ACM MM 2020**) [[paper](https://arxiv.org/pdf/2008.00767.pdf)][[code]( https://github.com/Ohraincu/DCSFN)][[homepage](https://github.com/Ohraincu)]

- **[CVID]** Conditional Variational Image Deraining (**TIP 2020**) [[paper](https://arxiv.org/pdf/2004.11373.pdf)][[code](https://github.com/Yingjun-Du/VID)][[homepage](https://github.com/Yingjun-Du)]

- **[DRD-Net]** Detail-recovery Image Deraining via Context Aggregation Networks (**CVPR 2020**) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_Detail-recovery_Image_Deraining_via_Context_Aggregation_Networks_CVPR_2020_paper.pdf)][[homepage](https://github.com/Dengsgithub)][[code](https://github.com/Dengsgithub/DRD-Net)]

- **[RCDNet]** A Model-driven Deep Neural Network for Single Image Rain Removal (**CVPR 2020**) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf)][[code](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Wang_A_Model-Driven_Deep_CVPR_2020_supplemental.pdf)]

- **[Syn2Rel]** Syn2Real Transfer Learning for Image Deraining using Gaussian Processes (**CVPR 2020**) [[paper](https://arxiv.org/pdf/2006.05580.pdf)][[homepage](https://github.com/rajeevyasarla)][[code](https://github.com/rajeevyasarla/Syn2Real)]

- **[MSPFN]** Multi-Scale Progressive Fusion Network for Single Image Deraining (**CVPR 2020**) [[paper](https://arxiv.org/pdf/2003.10985.pdf)][[code](https://github.com/kuihua/MSPFN)][[homepage](https://github.com/kuihua)]
   
- **[VID]** Variational Image Deraining (**WACV 2020**) [[paper](http://openaccess.thecvf.com/content_WACV_2020/papers/Du_Variational_Image_Deraining_WACV_2020_paper.pdf)][[homepage](https://csjunxu.github.io/)]

- **[CMGD]** (**TIP 2020**) [[paper](https://ieeexplore.ieee.org/abstract/document/9007569)][[homepage](https://github.com/rajeevyasarla)]


### 2019

- **Single Image Deraining: From Model-Based to Data-Driven and Beyond** (**TPAMI 2019**) [[paper](https://arxiv.org/pdf/1912.07150.pdf)]

- **[RWL]** Scale-Free Single Image Deraining Via VisibilityEnhanced Recurrent Wavelet Learning (**TIP 2019**) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8610325)]
 
- **A Survey on Rain Removal from Video and Single Image** [[paper](https://arxiv.org/pdf/1909.08326.pdf)][[code](https://github.com/hongwang01/Video-and-Single-Image-Deraining)]

- **[ERL-Net]**  ERL-Net: Entangled Representation Learning for Single Image De-Raining (**ICCV 2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_ERL-Net_Entangled_Representation_Learning_for_Single_Image_De-Raining_ICCV_2019_paper.pdf)][[code](https://github.com/RobinCSIRO/ERL-Net-for-Single-Image-Deraining)]

- **[ReHEN]** Single Image Deraining via Recurrent Hierarchy and Enhancement Network (**ACM MM 2019**) [[paper](http://delivery.acm.org/10.1145/3360000/3351149/p1814-yang.pdf?ip=202.120.235.180&id=3351149&acc=OPEN&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1573634982_715c64cb335fa08b82d82225f1944231#URLTOKEN#)][[code](https://github.com/nnUyi/ReHEN)][[homepage](https://nnuyi.github.io/)]

- **[DTDN]** DTDN: Dual-task De-raining Network (**ACM MM 2019**) [[paper](http://delivery.acm.org/10.1145/3360000/3350945/p1833-wang.pdf?ip=202.120.235.223&id=3350945&acc=OPEN&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1572964912_ad2b0e3c2bc1fdb6f216a99468d1a0ea#URLTOKEN#)]
  
- **[GraNet]** Gradual Network for Single Image De-raining (**ACM MM 2019**) [[paper](http://delivery.acm.org/10.1145/3360000/3350883/p1795-yu.pdf?ip=202.120.235.223&id=3350883&acc=OPEN&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1572964981_badf5608c2c0c67afa35ba86f50fe968#URLTOKEN#)]

- **[Dual-ResNet]** Dual Residual Networks Leveraging the Potential of Paired Operations for Image Restoration (**CVPR 2019**) [[paper](https://arxiv.org/pdf/1903.08817v1.pdf)][[code](https://github.com/liu-vis/DualResidualNetworks)]

- **[Heavy Rain Image Restoration]** Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning (**CVPR 2019**) [[paper](http://export.arxiv.org/pdf/1904.05050)][[code](https://github.com/liruoteng/HeavyRainRemoval)][[dataset](https://drive.google.com/file/d/1rFpW_coyxidYLK8vrcfViJLDd-BcSn4B/view)]

- **[SPANet]** Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset (**CVPR 2019**) [[paper](https://arxiv.org/pdf/1904.01538.pdf)][[code](https://github.com/stevewongv/SPANet)][[homepage](https://stevewongv.github.io/derain-project.html)][[dataset](https://stevewongv.github.io/derain-project.html)]

- **[Comprehensive Benchmark Analysis]** Single Image Deraining: A Comprehensive Benchmark Analysis (**CVPR 2019**) [[paper](https://arxiv.org/pdf/1903.08558.pdf)][[code](https://github.com/lsy17096535/Single-Image-Deraining)][[dataset](https://github.com/lsy17096535/Single-Image-Deraining)]

- **[DAF-Net]** Depth-attentional Features for Single-image Rain Removal (**CVPR 2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)][[code](https://github.com/xw-hu/DAF-Net)][[homepage](https://xw-hu.github.io/)]

- **[Semi-supervised Transfer Learning]** Semi-supervised Transfer Learning for Image Rain Removal (**CVPR 2019**) [[paper](https://arxiv.org/pdf/1807.11078.pdf)][[code](https://github.com/wwzjer/Semi-supervised-IRR)]

- **[PReNet]** Progressive Image Deraining Networks: A Better and Simpler Baseline (**CVPR 2019**) [[paper](https://arxiv.org/pdf/1901.09221.pdf)][[code](https://github.com/csdwren/PReNet)]

- **[UMRL-using-Cycle-Spinning]** Uncertainty Guided Multi-Scale Residual Learning-using a Cycle Spinning CNN for Single Image De-Raining (**CVPR 2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yasarla_Uncertainty_Guided_Multi-Scale_Residual_Learning-Using_a_Cycle_Spinning_CNN_for_CVPR_2019_paper.pdf)][[code](https://github.com/rajeevyasarla/UMRL--using-Cycle-Spinning)][[homepage](https://github.com/rajeevyasarla)]

- **[RR-GAN]** RR-GAN: Single Image Rain Removal Without Paired Information (**AAAI 2019**) [[paper](http://vijaychan.github.io/Publications/2019_derain.pdf)]
  
- **[D3R-Net]** D3R-Net: Dynamic Routing Residue Recurrent Network for Video Rain Removal (**TIP 2019**) [[paper](http://www.icst.pku.edu.cn/struct/Pub%20Files/2019/ywh_tip19.pdf)]

### 2018

- **[GCAN]** Gated Context Aggregation Network for Image Dehazing and Deraining (**WACV 2018**) [[paper](https://arxiv.org/pdf/1811.08747.pdf)][[code](https://github.com/cddlyf/GCANet)]
  
- **[RESCAN]** Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining (**ECCV 2018**) [[paper](https://arxiv.org/pdf/1807.05698.pdf)][[code](https://xialipku.github.io/RESCAN/)][[web](https://xialipku.github.io/RESCAN/)]

- **[RGFFN]** Residual-Guide Feature Fusion Network for Single Image Deraining (**ACM MM 2018**) [[paper](https://arxiv.org/abs/1804.07493)]

- **[NLEDN]** Non-locally Enhanced Encoder-Decoder Network for Single Image De-raining (**ACM MM 2018**) [[paper](https://arxiv.org/pdf/1808.01491.pdf)][[code](https://github.com/AlexHex7/NLEDN)]
    
- **[DualCNN]** Learning Dual Convolutional Neural Networks for Low-Level Vision (**CVPR 2018**) [[paper](http://faculty.ucmerced.edu/mhyang/papers/cvpr2018_dual_cnn.pdf)][[code](https://sites.google.com/site/jspanhomepage/dualcnn)][[web](https://sites.google.com/site/jspanhomepage/dualcnn)]
  
- **[Attentive GAN]**  Attentive Generative Adversarial Network for Raindrop Removal from a Single Image (**CVPR 2018**) (*tips: this research focuses on reducing the effets form the adherent rain drops instead of rain streaks removal*) [[paper](https://arxiv.org/abs/1711.10098)][[code]](https://github.com/rui1996/DeRaindrop) [[homepage]](https://rui1996.github.io/) [[project]](https://rui1996.github.io/raindrop/raindrop_removal.html) [[reimplement code](https://github.com/MaybeShewill-CV/attentive-gan-derainnet)]

- **[DID-MDN]** Density-aware Single Image De-raining using a Multi-stream Dense Network (**CVPR 2018**) [[paper](https://arxiv.org/abs/1802.07412)][[code](https://github.com/hezhangsprinter/DID-MDN)][[homepage]](https://sites.google.com/site/hezhangsprinter/) 
  
- **[Directional global sparse model]** A directional global sparse model for single image rain removal (**ACM MM 2018**) [[paper](https://www.sciencedirect.com/science/article/pii/S0307904X18301069)][[code](http://www.escience.cn/system/file?fileId=98760)][[homepage](http://www.escience.cn/people/dengliangjian/index.html)]

- **[MSCSC]** Video Rain Streak Removal By Multiscale ConvolutionalSparse Coding (**CVPR 2018**) [[paper](https://pan.baidu.com/s/1iiRr7ns8rD7sFmvRFcxcvw)][[code](https://github.com/MinghanLi/MS-CSC-Rain-Streak-Removal)][[homepage](https://sites.google.com/view/cvpr-anonymity)][[video](https://www.youtube.com/watch?v=tYHX7q0yK4M)]

- **[CNN Framework]** Robust Video Content Alignment and Compensation for Rain Removal in a CNN Framework (**CVPR 2018**) [[paper](https://arxiv.org/abs/1803.10433)][[homepage Chen](https://github.com/hotndy/SPAC-SupplementaryMaterials)][[homepage Chau](http://www.ntu.edu.sg/home/elpchau/)]

- **[Erase or Fill]** Erase or Fill? Deep Joint Recurrent Rain Removal and Reconstruction in Videos (**CVPR 2018**) [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Erase_or_Fill_CVPR_2018_paper.pdf)][[code](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018)][[homepage Liu]](http://www.icst.pku.edu.cn/struct/people/liujiaying.html) [[homepage Yang](http://www.icst.pku.edu.cn/struct/people/whyang.html)]

### 2017

- **[Transformed Low-Rank Model]** Transformed Low-Rank Model for Line Pattern Noise Removal (**ICCV 2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Chang_Transformed_Low-Rank_Model_ICCV_2017_paper.html)]

- **[JBO]** Joint Bi-layer Optimization for Single-image Rain Streak Removal (**ICCV 2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.html)][[homepage](http://appsrv.cse.cuhk.edu.hk/~lzhu/)]

- **[JCAS]** Joint Convolutional Analysis and Synthesis Sparse Representation for Single Image Layer Separation (**ICCV 2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Gu_Joint_Convolutional_Analysis_ICCV_2017_paper.html)][[code](http://www4.comp.polyu.edu.hk/~cslzhang/code/JCAS_Release.zip)][homepage](https://sites.google.com/site/shuhanggu/home)]

- **[DDN]** Removing rain from single images via a deep detail network (**CVPR 2017**) [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf) [[code](https://xueyangfu.github.io/projects/cvpr2017.html)][[homepage](https://xueyangfu.github.io/projects/cvpr2017.html)]
  
- **[JORDER]** Deep joint rain detection and removal from a single image (**CVPR 2017**) [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf) [[code](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)][[homepage](http://www.icst.pku.edu.cn/struct/people/whyang.html)]
 
- **[Hierarchical Approach]** A Hierarchical Approach for Rain or Snow Removing in a Single Color Image (**TIP 2017**) [[paper](http://ieeexplore.ieee.org/abstract/document/7934435/)]

- **[Clearing The Skies]** Clearing the skies: A deep network architecture for single-image rain removal (**TIP 2017**) [[paper](https://ieeexplore.ieee.org/abstract/document/7893758/)][[code](https://xueyangfu.github.io/projects/tip2017.html)][[homepage](https://xueyangfu.github.io/projects/tip2017.html)]

- **[MoG]** Should We Encode Rain Streaks in Video as Deterministic or Stochastic? (**ICCV 2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Wei_Should_We_Encode_ICCV_2017_paper.html)][[code](https://github.com/wwxjtu/RainRemoval_ICCV2017)][[homepage](https://github.com/wwxjtu/RainRemoval_ICCV2017)]

- **[FastDeRain]** A novel tensor-based video rain streaks removal approach via utilizing discriminatively intrinsic priors (**CVPR 2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Jiang_A_Novel_Tensor-Based_CVPR_2017_paper.html)][[code](https://github.com/TaiXiangJiang/FastDeRain)]

- **[Matrix Decomposition]** Video Desnowing and Deraining Based on Matrix Decomposition (**CVPR 2017**) [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Ren_Video_Desnowing_and_CVPR_2017_paper.html)]

### 2015-2016

- **[LP(GMM)]** Rain streak removal using layer priors (**CVPR 2016**) [[paper]](https://ieeexplore.ieee.org/document/7780668/) Single Image Rain Streak Decomposition Using Layer Priors (**TIP 2017**) [[dataset]](http://yu-li.github.io/paper/li_cvpr16_rain.zip) [[homepage]](http://yu-li.github.io/)

- **[DSC]** Removing rain from a single image via discriminative sparse coding (**ICCV 2016**) [[paper](http://ieeexplore.ieee.org/document/7410745/)][[code](http://www.math.nus.edu.sg/~matjh/download/image_deraining/rain_removal_v.1.1.zip)]

- **[Window Covered]** Restoring An Image Taken Through a Window Covered with Dirt or Rain (**ICCV 2013**) [[paper](https://cs.nyu.edu/~deigen/rain/)][[code](https://cs.nyu.edu/~deigen/rain/)]

- **[Image Decomposition]** Automatic Single-Image-Based Rain Streaks Removal via Image Decomposition (**TIP 2012**)[paper](http://www.ee.nthu.edu.tw/cwlin/Rain_Removal/tip_rain_removal_2011.pdf)][[code](http://www.ee.nthu.edu.tw/~cwlin/pub.htm)]
  
- **[Adherent Raindrop Modeling]** Adherent raindrop modeling, detectionand removal in video (**TPAMI 2016**) [[paper](https://ieeexplore.ieee.org/abstract/document/7299675/)][[homepage](http://www.cvl.iis.u-tokyo.ac.jp/~yousd/CVPR2013/Shaodi_CVPR2013.html)]

- **[Low-rank Matrix Completion]** Video deraining and desnowing using temporal correlation and low-rank matrix completion (**TIP 2015**) [[paper](https://ieeexplore.ieee.org/abstract/document/7101234/)][[code](http://mcl.korea.ac.kr/~jhkim/deraining/)]

- **[Utilizing Local Phase Information]** Utilizing local phase information to remove rain from video (**IJCV 2015**) [[paper](https://link.springer.com/article/10.1007/s11263-014-0759-8)]
