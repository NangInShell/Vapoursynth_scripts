import vapoursynth as vs
import havsfunc as haf
from vapoursynth import core

video = core.lsmas.LWLibavSource(r"D:\VS_test\隔行扫描测试.mov")

video = haf.QTGMC(video, Preset='Slower', TFF=True, FPSDivisor=2) #True OR False

#可以通过bicubic做后续的格式设置
#video = core.resize.Bicubic(video,format=vs.YUV420P8)

video.set_output()