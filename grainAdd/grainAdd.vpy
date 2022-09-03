from vapoursynth import core
import vapoursynth as vs
import havsfunc as haf
import mvsfunc as mvf
input = r'chunk-001-00000000-00000983.mkv'


src8   = core.lsmas.LWLibavSource(source=input)

src16  = mvf.Depth(src8,16,fulls=False,useZ=True)

noise = core.grain.Add(src16,50,50,0.4,0.4,2,True,0)
noise = core.std.ShufflePlanes(noise, planes=0, colorfamily=vs.GRAY)

output = core.std.ShufflePlanes([noise,src16], planes=[0, 1, 2], colorfamily=vs.YUV)
output.set_output()