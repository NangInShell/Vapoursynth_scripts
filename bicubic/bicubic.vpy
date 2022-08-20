import vapoursynth as vs
from vapoursynth import core

src = core.lsmas.LWLibavSource(r"D:\VS_test\NCED1.m2ts")#输入文件

src = core.resize.Bicubic(src,format=vs.YUV420P8,matrix=1,matrix_in=1)

src.set_output()


