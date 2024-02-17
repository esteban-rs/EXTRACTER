# EXTRACTER

Implementation of [Efficient Texture Matching with Attention and Gradient Enhancing for Large Scale Image Super Resolution](https://paperswithcode.com/paper/extracter-efficient-texture-matching-with)



Our pretrained models are avaiable at [Extracter-rec-pretrained](https://drive.google.com/file/d/1n8jLzbQUxWUCmuw2sz17IJa-ubUonCAw/view?usp=sharing) and [Extracter-All-Losses-pretrained](https://drive.google.com/file/d/1C0xT8vJWqdaz8IdtYd8xuxYD3rTAmzNf/view?usp=sharing)


# EXTRACTER

Implementation of [Efficient Texture Matching with Attention and Gradient Enhancing for Large Scale Image Super Resolution](https://paperswithcode.com/paper/extracter-efficient-texture-matching-with)


## Abstract
Recent Reference-Based image super-resolution (RefSR) has improved SOTA deep methods by introducing attention mechanisms to enhance low-resolution(LR) images by transferring high-resolution textures from a high-resolution
reference image. The main idea is to search for matches between patches using LR and Reference image pairs in a feature space and merge them using deep architectures. However, existing methods lack the accurate and efficient search
of textures. They divide images into as many patches as possible, resulting in inefficient memory usage, and cannot manage inference using large images. Herein, we propose a deep search that can dynamically change the window and stride size in the inference mode resulting in a more efficient memory usage that reduces significantly the number of image patches and finds the k most relevant texture match for each low-resolution patch over the high-resolution reference patches, resulting in an accurate texture match. Our main contribution is that, using larger kernels, we reduce the multiplication cost between LR and Reference features maintaining the SoTA performance and allowing us to generate large-dimension images (1024 px) in a single GPU. We enhance the Super Resolution result by adding gradient density information using a simple residual architecture showing competitive metrics results: PSNR and SSIM.

### Approach
<img src="https://github.com/esteban-rs/EXTRACTER/master/IMG/model1.png" width=40%><img src="https://github.com/esteban-rs/EXTRACTER/master/IMG/model2.png" width=60%>

### Main results
<img src="https://github.com/esteban-rs/EXTRACTER/master/IMG/results.png" width=80%>

## Requirements and dependencies
* python 3.7 
* pytorch >= 1.1.0
* torchvision >= 0.4.0

## Clone
1. Clone this github repo
```
git clone https://github.com/esteban-rs/EXTRACTER.git
cd EXTRACTER
```
2. Download pre-trained models
3. Run notebooks

4. Results are in "args.save_dir"
5. 
## Dataset prepare
1. Download [CUFED train set](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I) and [CUFED test set](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view)
2. Make dataset structure be:
- CUFED
    - train
        - input
        - ref
    - test
        - CUFED5

## Train
1. Prepare CUFED dataset
2. Run last cell in  01-Extracter-rec-v2.

## Citation
```
@misc{reyessaldana2023extracter,
      title={EXTRACTER: Efficient Texture Matching with Attention and Gradient Enhancing for Large Scale Image Super Resolution}, 
      author={Esteban Reyes-Saldana and Mariano Rivera},
      year={2023},
      eprint={2310.01379},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


