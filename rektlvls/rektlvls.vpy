import vapoursynth as vs
import mvsfunc as mvf
import havsfunc as haf
from rekt import rektlvls
core = vs.core

core.num_threads = 16
core.max_cache_size = 8000 

      
src = core.lsmas.LWLibavSource(r"D:\VS_test\NCED1.m2ts")

src = rektlvls(src,rownum=[1067,1068,1070,1071,1072,1073,1074,1075,1078], rowval=[-2,-4,4,0,-3,-10,-20,-10,15])

src.set_output()