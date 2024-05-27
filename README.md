# 羊毛纤维检测

系统是 **Ubuntu20.04** 的

需要 OpenCV 库 : OpenCV > 3.0

处理羊毛纤维图像，获取纤维的直径和直径变化度。如图所示
![原始图像](https://img-blog.csdnimg.cn/direct/6bc91bfb8134428bb35d9217d412094f.bmp#pic_center)

在 build 文件夹下面的终端，使用命令：

```bash
./wool ../pictures/3.bmp 
```

检测完成之后的结果如下：

![检测结果](https://img-blog.csdnimg.cn/direct/f89c6238a49d462ebe9f4d00b7c953c9.png#pic_center)

因为本身没有说要获取绝对世界下面的尺寸，而且咱也不知道具体的纤维尺寸是多少，因此这里的数据仅仅是参考的图像像素数据做的一个预估数值。


***如果有帮到你，麻烦在 Github 上面点一下 star 吧***
