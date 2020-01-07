# GAN

## artifacts
我们研究了常见的斑点状artifacts的起源，并发现生成器创建它们是为了规避其架构中的设计缺陷。我们重新设计了生成器中使用的normalization，从而删除了artifacts。


## 棋盘伪像
棋盘伪像的潜在原因之一，可以通过从反卷积切换到最近邻上采样，然后进行常规卷积来解决该问题。可以将`ConvTranspose2d`替换为以下图层：
```python
nn.Upsample(scale_factor=2, mode='bilinear'),
nn.ReflectionPad2d(1),
nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),
```

如果您训练足够⻓时间,有时棋盘工件会消失。也许您可以尝试训练更⻓的时间。