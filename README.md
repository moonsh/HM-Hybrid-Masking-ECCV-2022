## HM: Hybrid Masking for Few-Shot Segmentation
HM: Hybrid Masking for Few-Shot Segmentation

<p align="middle">
    <img src="figure/main_fig2.png">
</p>

## Performance



## Scripts

You just need to add the below lines to VAT and HSNet script.

            supprot_img2 = torch.zeros_like(support_img)            
            supprot_img2[:,0,:,:]= support_img[:,0,:,:]*support_mask 
            supprot_img2[:,1,:,:]= support_img[:,1,:,:]*support_mask  
            supprot_img2[:,2,:,:]= support_img[:,2,:,:]*support_mask  

            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            Input_masking = self.extract_feats(supprot_img2, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

            support_feats = self.mask_feature(support_feats, support_mask.clone())

            for i in range(len(support_feats)):
                s_r = torch.where(support_feats[i]>0, support_feats[i],  Input_masking[i] )
                support_feats[i] = s_r
                


            query_feats = self.resize_feats(query_feats, self.stack_ids)
            support_feats = self.resize_feats(support_feats, self.stack_ids)



For your convenience, we provide example of file. vat.py and hsnet.py


## Pretrained models

You can test using the pretrained models.

VAT [Link] 

Pascal-5 Benchmark with ResNet50

Pascal-5 Benchmark with ResNet101

COCO-20 Benchmark with ResNet50

FSS-1000 Benchmark with ResNet50

FSS-1000 Benchmark with ResNet101


HSNet [Link]

Pascal-5 Benchmark with ResNet50

Pascal-5 Benchmark with ResNet101

COCO-20 Benchmark with ResNet50

COCO-20 Benchmark with ResNet101

FSS-1000 Benchmark with ResNet50

FSS-1000 Benchmark with ResNet101


## References

We used works from [HSNet](https://github.com/juhongm999/hsnet), and [VAT](https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer). Thank you very much.

### BibTeX
If you find this research useful, please consider citing:

````BibTeX
@article{HMFS,
  title={HM: Hybrid Masking for Few-Shot Segmentation},
  author={Seonghyeon Moon, Sam Sohn},
  journal={arXiv preprint arXiv:2206.09667},
  year={2022}
}
````
