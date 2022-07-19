## HM: Hybrid Masking for Few-Shot Segmentation

<p align="middle">
    <img src="figure/main_fig2.png" width="600" height="350" />
</p>


## Scripts
This work can be implemented very easily by using the below script. 
The below script needs to be added to the [HSNet](https://github.com/juhongm999/hsnet), [VAT](https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer) and [ASNet](https://github.com/dahyun-kang/ifsl).

            supprot_img_im = torch.zeros_like(support_img)            
            supprot_img_im[:,0,:,:]= support_img[:,0,:,:]*support_mask 
            supprot_img_im[:,1,:,:]= support_img[:,1,:,:]*support_mask  
            supprot_img_im[:,2,:,:]= support_img[:,2,:,:]*support_mask  

            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            Input_masking = self.extract_feats(supprot_img_im, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

            support_feats = self.mask_feature(support_feats, support_mask.clone())

            for i in range(len(support_feats)):
                s_r = torch.where(support_feats[i]>0, support_feats[i],  Input_masking[i] )
                support_feats[i] = s_r
                
                
            query_feats = self.resize_feats(query_feats, self.stack_ids)           
            support_feats = self.resize_feats(support_feats, self.stack_ids)



For your convenience, we provide example of file. hsnet, vat.py and asnet.py


## Evaluation

Follow the testing direction for each method and use the pretrained models with the above script.

HSNet-HM [Link]
- Pascal-5 Benchmark with ResNet50
- Pascal-5 Benchmark with ResNet101
- COCO-20 Benchmark with ResNet50
- COCO-20 Benchmark with ResNet101
- FSS-1000 Benchmark with ResNet50
- FSS-1000 Benchmark with ResNet101

VAT-HM [Link]

- Pascal-5 Benchmark with ResNet50
- Pascal-5 Benchmark with ResNet101
- COCO-20 Benchmark with ResNet50
- FSS-1000 Benchmark with ResNet50
- FSS-1000 Benchmark with ResNet101

ASNet-HM [Link]

- COCO-20 Benchmark with ResNet50
- COCO-20 Benchmark with ResNet101

## Performance



## Visualization

<p align="middle">
    <img src="figure/comparison.png" width="600" height="350" />
</p>


## References

We used works from HSNet, VAT, and ASNet.

- [HSNet](https://github.com/juhongm999/hsnet) : Hypercorrelation Squeeze for Few-Shot Segmentation, 2021 ICCV
- [VAT](https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer) : Cost Aggregation with 4D Convolutional Swin Transformer for Few-Shot Segmentation, ECCV 2022
- [ASNet](https://github.com/dahyun-kang/ifsl) : Integrative Few-Shot Learning for Classification and Segmentation, CVPR 2022

Thank you very much.

### BibTeX
If you find this research useful, please consider citing:

````BibTeX
@article{HMFS,
  title={HM: Hybrid Masking for Few-Shot Segmentation},
  author={Seonghyeon Moon, Samuel S. Sohn, Honglu Zhou, Sejong Yoon, Vladimir Pavlovic, Muhammad Haris Khan, Mubbasir Kapadia},
  journal={arXiv preprint arXiv:2203.12826},
  year={2022}
}
````
