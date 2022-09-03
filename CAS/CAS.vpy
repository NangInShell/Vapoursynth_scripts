import vapoursynth as vs
import mvsfunc as mvf
import havsfunc as haf

core = vs.core

core.num_threads = 16
core.max_cache_size = 8000 

      
src = core.lsmas.LWLibavSource(r"D:\VS_test\NCED1.m2ts")

src=core.cas.CAS(src, sharpness=1)

src.set_output()