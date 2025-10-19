# SA<sup>2</sup>Net: Scale-Adaptive Structure-Affinity Transformation for Spine Segmentation from Ultrasound Volume Projection Imaging 🩻

> _[Hao Xie](https://scholar.google.com/citations?user=pDmY7YwAAAAJ&hl), [Zixun Huang](https://jacksonhuang.site/about_me/), [Yushen Zuo](https://yushenzuo.github.io/), [Yakun Ju](https://juyakun.github.io/), et al., Computerized Medical Imaging and Graphics, 2025_

## 🧠 Overview

> __Abstract__: Spine segmentation, based on ultrasound volume projection imaging (VPI), plays a vital role for intelligent scoliosis
> diagnosis in clinical applications. However, this task faces several significant challenges. Firstly, the global contextual
> knowledge of spines may not be well-learned if we neglect the high spatial correlation of different bone features.
> Secondly, the spine bones contain rich structural knowledge regarding their shapes and positions, which deserves
> to be encoded into the segmentation process. To address these challenges, we propose a novel
> scale-adaptive structure-aware network (SA<sup>2</sup>Net) for effective spine segmentation. First, we propose a scale-adaptive complementary strategy
> to learn the cross-dimensional long-distance correlation features for spinal images. Second, motivated by the consistency
> between multi-head self-attention in Transformers and semantic level affinity, we propose structure-affinity transformation
> to transform semantic features with class-specific affinity and combine it with a Transformer decoder for structure-aware reasoning.
> In addition, we adopt a feature mixing loss aggregation method to enhance model training.
> This method improves the robustness and accuracy of the segmentation process. The experimental results demonstrate that
> our SA<sup>2</sup>Net achieves superior segmentation performance compared to other state-of-the-art methods. Moreover,
> the adaptability of SA<sup>2</sup>Net to various backbones enhances its potential as a promising tool for
> advanced scoliosis diagnosis using intelligent spinal image analysis.

<p align="center">
  <img width="546" height="233" alt="framework" src="https://github.com/user-attachments/assets/21aae817-5ae2-48aa-9734-3e585c4fe1bd" />
  <br>
  <em>Overall framework of SA<sup>2</sup>Net</em>
</p>

## 🧩 Usage

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/taetiseo09/SA2Net.git
cd SA2Net

# 2. Create environment
conda create -n sa2net python=3.10
conda activate sa2net

# 3. Install dependencies
pip install -r requirements.txt
```

### Data preparation

Place all the datasets in one directory under ```data/```. The file structure looks like

```
data/
└──scoliosis3classes/
   ├──dataset/
   │  ├──train/
   │  └──test/
   ├──groundtruth/
   │  ├──train/
   │  └──test/
   ├──train.txt
   └──test.txt
```

We provide one demo figure about scoliosis for the evaluation.

### Training

```
python tools/train.py --work-dir ./Checkpoints
```
This will create the folder in current directory ```/Checkpoints``` to save logs and models.

### Inference

```
python tools/inference.py --config ./configs/... --Checkpoint ./Checkpoints/latest.pth
```

Run ```tools/inference.py``` to do evaluation of spine segmentation performance. Please specify at least one operation (save/eval/format/show the results / save the results) with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir".

## 🙏 Acknowledgements

- [**MMSegmentation**](https://github.com/open-mmlab/mmsegmentation): We follow the branch _0.x_ of MMSegmentation with a version of _0.30.0_ and implement the code of SA<sup>2</sup>Net.

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{xie2025sa2net,
  title     = {SA2Net: Scale-adaptive structure-affinity transformation for spine segmentation from ultrasound volume projection imaging},
  author    = {Xie, Hao and Huang, Zixun and Zuo, Yushen and Ju, Yakun and Leung, Frank HF and Law, NF and Lam, Kin-Man and Zheng, Yong-Ping and Ling, Sai Ho},
  journal   = {Computerized Medical Imaging and Graphics},
  year      = {2025},
  volume    = {125},
  pages     = {102649},
  publisher = {Elsevier}
}
```





