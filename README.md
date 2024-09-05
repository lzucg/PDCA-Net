# PDCA-Net
This is the implementation of our paper PDCA-Net: Parallel dual-channel attention network for polyp segmentation that has been submitted to Biomedical Signal Processing and Control.
# Description
Accurate segmentation of polyps in colonoscopy images plays a crucial role in the diagnosis and cure of colorectal cancer. Various deep learning methods have been applied to address the localization and precise segmentation of polyp regions, and have shown promising performance. However, the precise distinction between polyp and mucosal boundaries remains challenging, and accurate polyp segmentation remains to be an open problem. To address these challenges, we propose a Parallel Dual-Channel Attention Network (PDCA-Net) for polyp segmentation. This method utilizes the mapping transformations to adaptively encapsulate the global dependency from superpixel into pixels, enhancing the model’s ability to localize foreground and background regions. Specifically, we first design a parallel spatial and channel attention fusion module to capture the global dependencies at the superpixel level from the spatial and channel dimensions. Furthermore, an adaptive associative mapping module is proposed to encapsulate the global dependencies of superpixels into each pixel through a coarse-to-fine learning strategy. Extensive experiments have been conducted on four typical polyp segmentation datasets (ETIS, Kvasir-SEG, CVC-Clinic, CVC-Colon) to evaluate the segmentation performance of the proposed PDCA-Net. The experimental results demonstrate the effectiveness and superiority of our method.
## Environment setup
```
Python 3.7 +
torch 1.7.0
torchvision 0.8.0
scipy 1.7.3
tqdm 4.64.0
```
# Please download the following datasets: 
```
ETIS, Kvasir-SEG, CVC-Clinic, CVC-Colon
```
## Citation
This code is based on the [Comprehensive and Delicate: An Efficient Transformer for Image Restoration].
If you find this repository useful for your research, please consider citing the following paper(s):
```bibtex
@inproceedings{zhao2023comprehensive,
  title={Comprehensive and delicate: An efficient transformer for image restoration},
  author={Zhao, Haiyu and Gou, Yuanbiao and Li, Boyun and Peng, Dezhong and Lv, Jiancheng and Peng, Xi},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={14122--14132},
  year={2023}
}
```
For any enquiries, please contact the main authors.
