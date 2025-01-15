# Saliency-Guided No-Reference Omnidirectional Image Quality Assessment via Scene Content Perceiving

PyTorch reimplemetation for the paper:

Saliency-Guided No-Reference Omnidirectional Image Quality Assessment via Scene Content Perceiving, Youzhi Zhang, Lifei Wan, Deyang Liu, Xiaofei Zhou,  Ping An, Caifeng Shan, TIM 2024 




## Data Preparation(e.g. OIQA database)

### Viewport and saliency map generation



1. Download OIQA database consisting 16 reference and 320 distorted OI

2. Transfer original ERP format OI into six cubemap format viewports
   ```
   cd equi2cubic
   ConvertCVIQtoCubic.m
   ```

3. Generate corresponding viewport saliency map. Firstly, we employ relative total variation (RTV)
  to extrtact the primary structures. Secondly, we utilize the robust back ground detection (RBD)
  method to generate a saliency map of each viewport
   ```
      cd ../RTV_Smooth-master/RTV_Smooth-master
      demo.m
      ```
   ```
      cd ../Saliency_Optimizaiton_Robust_background_2014/MCode_AddSLICSourceCode_8_30_2014
      demo.m
      ```

4. The processed saliency maps are provided. Download [Baidu Drive]():





## Testing
Our pretrained model on OIQA is provided on [Baidu Drive](). You can put it to `OQIA/final` 

    

## Citation
If you find this paper useful, please cite:
```
@ARTICLE{10731918,
  author={Zhang, Youzhi and Wan, Lifei and Liu, Deyang and Zhou, Xiaofei and An, Ping and Shan, Caifeng},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Saliency-Guided No-Reference Omnidirectional Image Quality Assessment via Scene Content Perceiving}, 
  year={2024},
  volume={73},
  number={},
  pages={1-15}}
```