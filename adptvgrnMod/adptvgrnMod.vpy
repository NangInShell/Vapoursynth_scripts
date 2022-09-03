from vapoursynth import core
import vapoursynth as vs
import mvsfunc as mvf

from adptvgrnMod import adptvgrnMod


src   = core.lsmas.LWLibavSource(r"D:\VS_test\test.mp4")

#有很多参数可以自己调整
noise = adptvgrnMod(src, size=10, strength=[10,10], sharp=100, luma_scaling=50, seed=3, show_mask=0)

noise.set_output()