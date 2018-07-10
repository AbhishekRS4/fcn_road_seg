# FCN implementation for road segmentation on camvid dataset

* Pretrained vgg-16 encoder is used
* Decoder is trained [fcn-8, fcn-16, fcn-32] for road segmentation

## References
* [Camvid Dataset](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/)
* [VGG for image classification](https://arxiv.org/pdf/1409.1556.pdf)
* [FCN for semantic segmentation](https://arxiv.org/pdf/1605.06211.pdf)
* Loss based on dice similarity score - [Dice loss for semantic segmentation](https://arxiv.org/pdf/1803.11078.pdf)
* FCN has also been used in [MultiNet](https://arxiv.org/pdf/1612.07695.pdf)

## To do
- [x] Camvid dataloader 
- [x] Pretrained vgg-16 encoder
- [x] Different losses
- [x] Different FCN decoders
- [x] Performance comparison

## Performance comparison
|model |      loss         | train accuracy | validation accuracy | test accuracy | 
|------|-------------------|----------------|---------------------|---------------|
|fcn-8 |   cross-entropy   |     97.431     |        95.328       |     96.030    |
|      |   dice-loss       |     97.471     |        95.762       |     96.204    |
|      |   combined        |     97.554     |        95.454       |     96.159    |
|fcn-16|   cross-entropy   |     97.458     |        95.654       |     96.105    |
|      |   dice-loss       |     97.456     |        95.439       |     96.110    |
|      |   combined        |     97.509     |        95.685       |     96.199    |
|fcn-32|   cross-entropy   |     97.166     |        95.547       |     96.056    |
|      |   dice-loss       |     97.141     |        95.401       |     95.784    | 
|      |   combined        |     97.182     |        95.449       |     95.947    |

* The metric used for comparison is pixel-wise accuracy score
