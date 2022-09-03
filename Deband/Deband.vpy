import vapoursynth as vs
import mvsfunc as mvf
import havsfunc as haf

core = vs.core

core.num_threads = 16
core.max_cache_size = 8000 

      
src = core.lsmas.LWLibavSource(r"D:\VS_test\NCED1.m2ts")

src  = core.neo_f3kdb.Deband(src,preset="medium",output_depth=16)

src.set_output()