# FCN implementation for road segmentation on camvid dataset

* [Camvid Dataset](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/)
* [FCN for semantic segmentation](https://arxiv.org/abs/1411.4038)
* Pretrained vgg-16 encoder is used
* Decoder is trained [fcn-8, fcn-16, fcn-32] only for road segmentation

## To do
- [x] Camvid dataloader 
- [x] Pretrained vgg-16 encoder code
- [x] Different losses
- [x] Decoder network
- [ ] Performance comparison
