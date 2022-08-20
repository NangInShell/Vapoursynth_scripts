# Bicubic参数解析

```python
resize.Bicubic(vnode clip[, int width, int height, int format, enum matrix, enum transfer, enum primaries, enum range, enum chromaloc, enum matrix_in, enum transfer_in, enum primaries_in, enum range_in, enum chromaloc_in, float filter_param_a, float filter_param_b, string resample_filter_uv, float filter_param_a_uv, float filter_param_b_uv, string dither_type="none", string cpu_type, bint prefer_props=False, float src_left, float src_top, float src_width, float src_height, float nominal_luminance])
```

以下参考官方文档的机器翻译和个人理解,仅供参考.有不对的地方欢迎指出

**clip**:视频流，适配各种类型的输入格式。
**width**:宽度
**height**:高度
**format**:输出格式，参数如下。使用时候写为类似vs.YUV420P8这样的格式。
GRAY8 YUV420P8 YUV420P9 YUV420P10 YUV420P12 YUV420P14 YUV420P16 YUV444PH RGB24 RGB36 RGBH
GRAY9 YUV422P8 YUV422P9 YUV422P10 YUV422P12 YUV422P14 YUV422P16 YUV444PS  RGB27 RGB42 RGBS
GRAY10 YUV444P8 YUV444P9 YUV444P10 YUV444P12 YUV444P14 YUV444P16                   RGB30 RGB48
GRAY12 YUV410P8
GRAY14 YUV411P8 
GRAY16 YUV440P8
GRAY32
GRAYH
GRAYS

**matrix, transfer, primaries**:指定色彩空间。未指定的话选择输入的clip的相应属性，其中YCoCg 和 RGB的色彩系列除外，其中对应的色彩矩阵是默认的。
**Matrix参数**：

rgb (0) ,709 (1) ,unspec (2) ,fcc (4),470bg (5),170m (6) ,240m(7), ycgco (8) ,2020ncl (9),2020cl (10) ,chromancl (12),chromacl (13),ictcp (14) .
**Transfer参数**：

709 (1),unspec (2),470m (4), 470bg (5),601 (6),240m (7),linear (8),log100 (9),log316 (10),xvycc (11),srgb (13),2020_10 (14),2020_12 (15),st2084 (16),std-b67 (18).
**primaries参数**：

709 (1),unspec (2),470m (4),470bg (5),170m (6),240m (7),film (8),2020 (9),st428 (10),xyz (10),st431-2 (11),st432-1 (12),jedec-p22 (22)

**range**：输出像素范围。对于整数格式，这可以选择规定的代码值，但也可能生成超过范围的数值。如果输入格式属于不同的颜色系列，则 YUV 的默认范围为 studio/limited，RGB 的默认范围为全范围。参数有：limited，full

**chromaloc**：输出色度位置。对于子采样格式，需要指定色度位置。如果输入格式为 4：4：4 或 RGB，并且输出已进行子采样，则根据 MPEG，默认位置为左对齐。 Possible chroma locations：left, center, top_left, top, bottom_left, bottom

**matrix_in、transfer_in、primaries_in、range_in chromaloc_in**：输入颜色空间/格式规范。如果将相应的 frame 属性设置为未指定的值以外的值，则使用原帧的属性而不是此参数。某些颜色空间会被设置为默认值。默认情况下，如果存在，它们将覆盖相应的帧属性，而是在两者都存在时为帧属性指定优先级，prefer_props。有关详细信息，请参阅上面的等效输出参数。（PS：我的建议是这些参数和上面对应的色域参数都填写完整）

**filter_param_a, filter_param_b**:用于 RGB 和 Y 通道的缩放器的参数。对于双三重滤波器，filter_param_a/b 表示“b”和“c”参数。对于 lanczos 滤波器，filter_param_a表示抽头次数。

**resample_filter_uv**：用于 UV 通道的缩放器的参数。它默认为与 Y 通道相同。以下值可用于resample_filter_uv：point, bilinear, bicubic, spline16, spline36, lanczos

**dither_type**：抖动方法。抖动仅用于生成整数格式的转换。可以使用以下抖动参数：none, ordered, random, error_diffusion

**cpu_type**：此功能现在仅用于测试

**prefer_props**：确定帧属性或参数在两者都存在时是否优先。此选项会影响matrix_in、transfer_in、primaries_in、range_in和chromaloc_in参数及其框架属性等效项。

**src_left、src_top、src_width src_height**：用于选择要使用的输入的源区域。也可用于移动图像。默认为整个图像。

**nominal_luminance**：确定值 1.0 的物理亮度。