import vapoursynth as vs
import mvsfunc as mvf
import havsfunc as haf

core = vs.core

core.num_threads = 16
core.max_cache_size = 8000 

      
src = core.lsmas.LWLibavSource(r"D:\VS_test\NCED1.m2ts")

src=core.knlm.KNLMeansCL(src,d=1,a=2,h=3,device_type='GPU',device_id=0)

src.set_output()