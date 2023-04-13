

# Vapoursynth变量与函数详解

[TOC]

## 一,变量

## 二,视频流相关函数

### **AddBorders**

```
std.AddBorders(vnode clip[, int left=0, int right=0, int top=0, int bottom=0, float[] color=<black>])

Adds borders to frames. The arguments specify the number of pixels to add on each side. They must obey the subsampling restrictions. The newly added borders will be set to color.
```

### **AssumeFPS**

```
std.AssumeFPS(vnode clip[, vnode src, int fpsnum, int fpsden=1])

Returns a clip with the framerate changed. This does not in any way modify the frames, only their metadata.

The framerate to assign can either be read from another clip, src, or given as a rational number with fpsnum and fpsden.

It is an error to specify both src and fpsnum.

AssumeFPS overwrites the frame properties _DurationNum and _DurationDen with the frame duration computed from the new frame rate.
```

### **AverageFrames**

```
std.AverageFrames(vnode[] clips, float[] weights[, float scale, bint scenechange, int[] planes])

AverageFrames has two main modes depending on whether one or multiple clips are supplied. The filter is named AverageFrames since using ones for weights is an easy way to average many frames together but it can also be seen as a temporal or multiple frame convolution.

If multiple clips are supplied then the frames from each of the clips are multiplied by the respective weights, summed together and divided by scale before being output. Note that only integer weights and scale are allowed for integer input formats.

If a single clip is supplied then an odd number of weights are needed and they will instead be temporally centered on the current frame of the clip. The rest works as multiple clip mode with the only difference being that scenechange can be set to avoid averaging frames over scene changes. If this happens then all the weights beyond a scene change are instead applied to the frame right before it.

At most 31 weights can be supplied.
```

### **Binarize/BinarizeMask**

```
std.Binarize(vnode clip[, float[] threshold, float[] v0, float[] v1, int[] planes=[0, 1, 2]])Á
std.BinarizeMask(vnode clip[, float[] threshold, float[] v0, float[] v1, int[] planes=[0, 1, 2]])Á

Turns every pixel in the image into either v0, if it’s below threshold, or v1, otherwise. The BinarizeMask version is intended for use on mask clips where all planes have the same value range and only differs in the default values of v0 and v1.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

threshold
Defaults to the middle point of range allowed by the format. Can be specified for each plane individually.

v0
Value given to pixels that are below threshold. Can be specified for each plane individually. Defaults to the lower bound of the format.

v1
Value given to pixels that are greater than or equal to threshold. Defaults to the maximum value allowed by the format. Can be specified for each plane individually. Defaults to the upper bound of the format.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.
```

### **BlankClip**

```
std.BlankClip([vnode clip, int width=640, int height=480, int format=vs.RGB24, int length=(10*fpsnum)/fpsden, int fpsnum=24, int fpsden=1, float[] color=<black>, bint keep=0])

Generates a new empty clip. This can be useful to have when editing video or for testing. The default is a 640x480 RGB24 24fps 10 second long black clip. Instead of specifying every property individually, BlankClip can also copy the properties from clip. If both an argument such as width, and clip are set, then width will take precedence.

If keep is set, a reference to the same frame is returned on every request. Otherwise a new frame is generated every time. There should usually be no reason to change this setting.

It is never an error to use BlankClip.
```

### **BoxBlur**

```
std.BoxBlur(vnode clip[, int[] planes, int hradius = 1, int hpasses = 1, int vradius = 1, int vpasses = 1])

Performs a box blur which is fast even for large radius values. Using multiple passes can be used to fairly cheaply approximate a gaussian blur. A radius of 0 means no processing is performed.
```

### **ClipToProp**

```
std.ClipToProp(vnode clip, vnode mclip[, string prop='_Alpha'])
Stores each frame of mclip as a frame property named prop in clip. This is primarily intended to attach mask/alpha clips to another clip so that editing operations will apply to both. Unlike most other filters the output length is derived from the second argument named mclip.

If the attached mclip does not represent the alpha channel, you should set prop to something else.

It is the inverse of PropToClip().
```

### **Convolution**

```
Convolutionstd.Convolution(vnode clip, float[] matrix[, float bias=0.0, float divisor=0.0, int[] planes=[0, 1, 2], bint saturate=True, string mode="s"])

Performs a spatial convolution.

Here is how a 3x3 convolution is done. Each pixel in the 3x3 neighbourhood is multiplied by the corresponding coefficient in matrix. The results of the nine multiplications are added together, then this sum is divided by divisor. Next, bias is added, and the result is rounded to the nearest larger integer. If this integer result is negative and the saturate parameter is False, it is multiplied by -1. Finally, the result is clamped to the format’s range of valid values.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

matrix
Coefficients for the convolution.

When mode is “s”, this must be an array of 9 or 25 numbers, for a 3x3 or 5x5 convolution, respectively.

When mode is not “s”, this must be an array of 3 to 25 numbers, with an odd number of elements.

The values of the coefficients must be between -1023 and 1023 (inclusive). The coefficients are rounded to integers when the input is an integer format.

This is how the elements of matrix correspond to the pixels in a 3x3 neighbourhood:

1 2 3
4 5 6
7 8 9
It’s the same principle for the other types of convolutions. The middle element of matrix always corresponds to the center pixel.

bias
Value to add to the final result of the convolution (before clamping the result to the format’s range of valid values).

divisor
Divide the output of the convolution by this value (before adding bias).

If this parameter is 0.0 (the default), the output of the convolution will be divided by the sum of the elements of matrix, or by 1.0, if the sum is 0.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.

saturate
The final result is clamped to the format’s range of valid values (0 .. (2**bitdepth)-1). Therefore, if this parameter is True, negative values become 0. If this parameter is False, it’s instead the absolute value that is clamped and returned.

mode
Selects the type of convolution. Possible values are “s”, for square, “h” for horizontal, “v” for vertical, and “hv” or “vh” for both horizontal and vertical.

How to apply a simple blur equivalent to Avisynth’s Blur(1):

Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
How to apply a stronger blur equivalent to Avisynth’s Blur(1.58):

Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
```

### **CopyFrameProps**

```
std.CopyFrameProps(vnode clip, vnode prop_src)

Returns clip but with all the frame properties replaced with the ones from the clip in prop_src. Note that if clip is longer than prop_src then the last existing frame’s properties will be used instead.
```

### **Crop/CropAbs**

```
std.Crop(vnode clip[, int left=0, int right=0, int top=0, int bottom=0])
std.CropAbs(vnode clip, int width, int height[, int left=0, int top=0])
Crops the frames in a clip.

Crop is the simplest to use of the two. The arguments specify how many pixels to crop from each side. This function used to be called CropRel which is still an alias for it.

CropAbs, on the other hand, is special, because it can accept clips with variable frame sizes and crop out a fixed size area, thus making it a fixed size clip.

Both functions return an error if the whole picture is cropped away, if the cropped area extends beyond the input or if the subsampling restrictions aren’t met.
```

### **Deflate/Inflate**

```
std.Deflate(vnode clip[, int[] planes=[0, 1, 2], float threshold])
Replaces each pixel with the average of the eight pixels in its 3x3 neighbourhood, but only if that average is less than the center pixel.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.

threshold
Allows to limit how much pixels are changed. Output pixels will not become less than input - threshold. The default is no limit.
```

```
std.Inflate(vnode clip[, int[] planes=[0, 1, 2], int threshold=65535])

Replaces each pixel with the average of the eight pixels in its 3x3 neighbourhood, but only if that average is greater than the center pixel.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.

threshold
Allows to limit how much pixels are changed. Output pixels will not become greater than input + threshold. The default is no limit.
```

### **DeleteFrames**

```
std.DeleteFrames(vnode clip, int[] frames)
Deletes the specified frames.

All frame numbers apply to the input clip.

Returns an error if the same frame is deleted twice or if all frames in a clip are deleted.
```

### **DoubleWeave**

```
std.DoubleWeave(vnode clip[, bint tff])
Weaves the fields back together from a clip with interleaved fields.

Since VapourSynth only has a weak notion of field order internally, tff may have to be set. Setting tff to true means top fields first and false means bottom fields first. Note that the _Field frame property, if present and in a valid combination, takes precedence over tff.

DoubleWeave’s output has the same number of frames as the input. One must use DoubleWeave together with SelectEvery to undo the effect of SeparateFields:

sep = core.std.SeparateFields(source)
...
woven = core.std.DoubleWeave(sep)
woven = core.std.SelectEvery(woven, 2, 0)

The _Field frame property is deleted and _FieldBased is set accordingly.
```

### **DuplicateFrames**

```
std.DuplicateFrames(vnode clip, int[] frames)
Duplicates the specified frames.

A frame may be duplicated several times.

All frame numbers apply to the input clip.
```

### **Expr**

```
std.Expr(vnode[] clips, string[] expr[, int format])
Expr evaluates an expression per pixel for up to 26 input clips. The expression, expr, is written using reverse polish notation and can be specified for each plane individually. The expression given for the previous plane is used if the expr array contains fewer expressions than the input clip has planes. In practice this means that a single expression will be applied to all planes by default.

Specifying an empty string as the expression enables a fast plane copy from the first specified clip, when possible. If it is not possible due to the output format being incompatible, the plane contents will be undefined.

Since the expression is evaluated at runtime, there are a few pitfalls. In order to keep speed up, the input ranges are not normalized to the usual floating point ranges. Instead they are left as is, meaning that an 8 bit clip will have values in the 0-255 range and a 10 bit clip will have values in the 0-1023 range. Note that floating point clips are even more difficult, as most channels are stored in the 0-1 range with the exception of U, V, Co and Cg planes, which are in the -0.5-0.5 range. If you mix clips with different input formats this must be taken into consideration.

When the output format uses integer samples, the result of the expression is clamped to the [0, 2**bits_per_sample-1] range. When the output format uses float samples, the result of the expression is stored without any clamping.

By default the output format is the same as the first input clip’s format. You can override it by setting format. The only restriction is that the output format must have the same subsampling as the input clips and be 8..16 bit integer or 32 bit float. 16 bit float is also supported on cpus with the f16c instructions.

Logical operators are also a bit special, since everything is done in floating point arithmetic. All values greater than 0 are considered true for the purpose of comparisons. Logical operators return 0.0 for false and 1.0 for true in their operations.

Since the expression is being evaluated at runtime, there are also the stack manipulation operators, swap and dup. The former swaps the topmost and second topmost values, and the latter duplicates the topmost stack value.

These operators also have swapN and dupN forms that allow a value N steps up in the stack to be swapped or duplicated. The top value of the stack has index zero meaning that dup is equivalent to dup0 and swap is equivalent to swap1. This is because swapN always swaps with the topmost value at index 0.

Expressions are converted to byte-code or machine-code by an optimizing compiler and are not guaranteed to evaluate in the order originally written. The compiler assumes that all input values are finite (i.e neither NaN nor INF) and that no operator will produce a non-finite value. Such expressions are invalid. This is especially important for the transcendental operators:

exp - expression must not overflow (i.e. x <= 88)

log - input must be finite and non-negative (i.e. x >= 0 && x <= 3e+38)

pow - base must be finite and non-negative. Result must not overflow (i.e. x >= 0 && x <= 3e+38; 1e-38 <= result <= 3e+38)

Clip load operators:

x-z, a-w
The operators taking one argument are:

exp log sqrt sin cos abs not dup dupN
The operators taking two arguments are:

+ - * / max min pow > < = >= <= and or xor swap swapN
The operators taking three arguments are:

?
For example these operations:

a b c ?

d e <

f abs
Are equivalent to these operations in C:

a ? b : c

d < e

abs(f)
The sin/cos operators are approximated to within 2e-6 absolute error for inputs with magnitude up to 1e5, and there is no accuracy guarantees for inputs whose magnitude is larger than 2e5.

How to average the Y planes of 3 YUV clips and pass through the UV planes unchanged (assuming same format):

std.Expr(clips=[clipa, clipb, clipc], expr=["x y + z + 3 /", "", ""])
How to average the Y planes of 3 YUV clips and pass through the UV planes unchanged (different formats):

std.Expr(clips=[clipa16bit, clipb10bit, clipa8bit],
   expr=["x y 64 * + z 256 * + 3 /", ""])
Setting the output format because the resulting values are illegal in a 10 bit clip (note that the U and V planes will contain junk since direct copy isn’t possible):

std.Expr(clips=[clipa10bit, clipb16bit, clipa8bit],
   expr=["x 64 * y + z 256 * + 3 /", ""], format=vs.YUV420P16)
```

### **FlipVertical/FlipHorizontal**

```
std.FlipVertical(vnode clip)
std.FlipHorizontal(vnode clip)
Flips the clip in the vertical or horizontal direction.
```

### **FrameEval**

```
FrameEvalstd.FrameEval(vnode clip, func eval[, vnode[] prop_src, vnode[] clip_src])
Allows an arbitrary function to be evaluated every frame. The function gets the frame number, n, as input and should return a clip the output frame can be requested from.

The clip argument is only used to get the output format from since there is no reliable automatic way to deduce it.

When using the argument prop_src the function will also have an argument, f, containing the current frames. This is mainly so frame properties can be accessed and used to make decisions. Note that f will only be a list if more than one prop_src clip is provided.

The clip_src argument only exists as a way to hint which clips are referenced in the eval function which can improve caching and graph generation. Its use is encouraged but not required.

This function can be used to accomplish the same things as Animate, ScriptClip and all the other conditional filters in Avisynth. Note that to modify per frame properties you should use ModifyFrame.

How to animate a BlankClip to fade from white to black. This is the simplest use case without using the prop_src argument:

import vapoursynth as vs
import functools

base_clip = vs.core.std.BlankClip(format=vs.YUV420P8, length=1000, color=[255, 128, 128])

def animator(n, clip):
   if n > 255:
      return clip
   else:
      return vs.core.std.BlankClip(format=vs.YUV420P8, length=1000, color=[n, 128, 128])

animated_clip = vs.core.std.FrameEval(base_clip, functools.partial(animator, clip=base_clip))
animated_clip.set_output()
How to perform a simple per frame auto white balance. It shows how to access calculated frame properties and use them for conditional filtering:

import vapoursynth as vs
import functools
import math

def GrayWorld1Adjust(n, f, clip, core):
   small_number = 0.000000001
   red   = f[0].props['PlaneStatsAverage']
   green = f[1].props['PlaneStatsAverage']
   blue  = f[2].props['PlaneStatsAverage']
   max_rgb = max(red, green, blue)
   red_corr   = max_rgb/max(red, small_number)
   green_corr = max_rgb/max(green, small_number)
   blue_corr  = max_rgb/max(blue, small_number)
   norm = max(blue, math.sqrt(red_corr*red_corr + green_corr*green_corr + blue_corr*blue_corr) / math.sqrt(3), small_number)
   r_gain = red_corr/norm
   g_gain = green_corr/norm
   b_gain = blue_corr/norm
   return core.std.Expr(clip, expr=['x ' + repr(r_gain) + ' *', 'x ' + repr(g_gain) + ' *', 'x ' + repr(b_gain) + ' *'])

def GrayWorld1(clip, matrix_s=None):
   rgb_clip = vs.core.resize.Bilinear(clip, format=vs.RGB24)
   r_avg = vs.core.std.PlaneStats(rgb_clip, plane=0)
   g_avg = vs.core.std.PlaneStats(rgb_clip, plane=1)
   b_avg = vs.core.std.PlaneStats(rgb_clip, plane=2)
   adjusted_clip = vs.core.std.FrameEval(rgb_clip, functools.partial(GrayWorld1Adjust, clip=rgb_clip, core=vs.core), prop_src=[r_avg, g_avg, b_avg])
   return vs.core.resize.Bilinear(adjusted_clip, format=clip.format.id, matrix_s=matrix_s)

vs.core.std.LoadPlugin(path='ffms2.dll')
main = vs.core.ffms2.Source(source='...')
main = GrayWorld1(main)
main.set_output()
```

### **FreezeFrames**

```
std.FreezeFrames(vnode clip, int[] first, int[] last, int[] replacement)
FreezeFrames replaces all the frames in the [first,last] range (inclusive) with replacement.

A single call to FreezeFrames can freeze any number of ranges:

core.std.FreezeFrames(input, first=[0, 100, 231], last=[15, 112, 300], replacement=[8, 50, 2])
This replaces [0,15] with 8, [100,112] with 50, and [231,300] with 2 (the original frame number 2, not frame number 2 after it was replaced with number 8 by the first range).

The frame ranges must not overlap.
```

### **Interleave**

```
std.Interleave(vnode[] clips[, bint extend=0, bint mismatch=0, bint modify_duration=True])

Returns a clip with the frames from all clips interleaved. For example, Interleave(clips=[A, B]) will return A.Frame 0, B.Frame 0, A.Frame 1, B.Frame…

The extend argument controls whether or not all input clips will be treated as if they have the same length as the longest clip.

Interleaving clips with different formats or dimensions is considered an error unless mismatch is true.

If modify_duration is set then the output clip’s frame rate is the first input clip’s frame rate multiplied by the number of input clips. The frame durations are divided by the number of input clips. Otherwise the first input clip’s frame rate is used.
```

### **Invert/InvertMask**

```
std.Invert(vnode clip[, int[] planes=[0, 1, 2]])
std.InvertMask(vnode clip[, int[] planes=[0, 1, 2]])

Inverts the pixel values. Specifically, it subtracts the value of the input pixel from the format’s maximum allowed value. The InvertMask version is intended for use on mask clips where all planes have the same maximum value regardless of the colorspace.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.
```

### **Levels**

```
std.Levels(vnode clip[, float min_in, float max_in, float gamma=1.0, float min_out, float max_out, int[] planes=[0, 1, 2]])
Adjusts brightness, contrast, and gamma.

The range [min_in, max_in] is remapped into [min_out, max_out]. Note that the range behavior is unintuitive for YUV float formats since the assumed range will be 0-1 even for the UV-planes.

For example, to convert from limited range YUV to full range (8 bit):

clip = std.Levels(clip, min_in=16, max_in=235, min_out=0, max_out=255, planes=0)
clip = std.Levels(clip, min_in=16, max_in=240, min_out=0, max_out=255, planes=[1,2])
The default value of max_in and max_out is the format’s minimum and maximum allowed values respectively. Note that all input is clamped to the input range to prevent out of range output.

Warning

The default ranges are 0-1 for floating point formats. This may have an undesired
effect on YUV formats.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

gamma
Controls the degree of non-linearity of the conversion. Values greater than 1.0 brighten the output, while values less than 1.0 darken it.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.
```

### **Limiter**

```
std.Limiter(vnode clip[, float[] min, float[] max, int[] planes=[0, 1, 2]])
Limits the pixel values to the range [min, max].

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

min
Lower bound. Defaults to the lowest allowed value for the input. Can be specified for each plane individually.

max
Upper bound. Defaults to the highest allowed value for the input. Can be specified for each plane individually.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.
```

### **Loop**

```
std.Loop(vnode clip[, int times=0])

Returns a clip with the frames or samples repeated over and over again. If times is less than 1 the clip will be repeated until the maximum clip length is reached, otherwise it will be repeated times times.

In Python, std.Loop can also be invoked using the multiplication operator.
```

### **Lut**

```
std.Lut(vnode clip[, int[] planes, int[] lut, float[] lutf, func function, int bits, bint floatout])

Applies a look-up table to the given clip. The lut can be specified as either an array of 2^bits_per_sample values or given as a function having an argument named x to be evaluated. Either lut, lutf or function must be used. The lut will be applied to the planes listed in planes and the other planes will simply be passed through unchanged. By default all planes are processed.

If floatout is set then the output will be floating point instead, and either lutf needs to be set or function always needs to return floating point values.

How to limit YUV range (by passing an array):

luty = []
for x in range(2**clip.format.bits_per_sample):
   luty.append(max(min(x, 235), 16))
lutuv = []
for x in range(2**clip.format.bits_per_sample):
   lutuv.append(max(min(x, 240), 16))
ret = Lut(clip=clip, planes=0, lut=luty)
limited_clip = Lut(clip=ret, planes=[1, 2], lut=lutuv)
How to limit YUV range (using a function):

def limity(x):
   return max(min(x, 235), 16)
def limituv(x):
   return max(min(x, 240), 16)
ret = Lut(clip=clip, planes=0, function=limity)
limited_clip = Lut(clip=ret, planes=[1, 2], function=limituv)
```

### **Lut2**

```
std.Lut2(vnode clipa, vnode clipb[, int[] planes, int[] lut, float[] lutf, func function, int bits, bint floatout])

Applies a look-up table that takes into account the pixel values of two clips. The lut needs to contain 2^(clip1.bits_per_sample + clip2.bits_per_sample) entries and will be applied to the planes listed in planes. Alternatively a function taking x and y as arguments can be used to make the lut. The other planes will be passed through unchanged. By default all planes are processed.

Lut2 also takes an optional bit depth parameter, bits, which defaults to the bit depth of the first input clip, and specifies the bit depth of the output clip. The user is responsible for understanding the effects of bit depth conversion, specifically from higher bit depths to lower bit depths, as no scaling or clamping is applied.

If floatout is set then the output will be floating point instead, and either lutf needs to be set or function always needs to return floating point values.

How to average 2 clips:

lut = []
for y in range(2 ** clipy.format.bits_per_sample):
   for x in range(2 ** clipx.format.bits_per_sample):
      lut.append((x + y)//2)
Lut2(clipa=clipa, clipb=clipb, lut=lut)
How to average 2 clips with a 10-bit output:

def f(x, y):
   return (x*4 + y)//2
Lut2(clipa=clipa8bit, clipb=clipb10bit, function=f, bits=10)
```

### **MakeDiff**

```
std.MakeDiff(vnode clipa, vnode clipb[, int[] planes])

Calculates the difference between clipa and clipb and clamps the result. By default all planes are processed. This function is usually used together with MergeDiff, which can be used to add back the difference.

Unsharp masking of luma:

blur_clip = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=[0])
diff_clip = core.std.MakeDiff(clip, blur_clip, planes=[0])
sharpened_clip = core.std.MergeDiff(clip, diff_clip, planes=[0])
```

### **MakeFullDiff**

```
std.MakeFullDiff(vnode clipa, vnode clipb)

Calculates the difference between clipa and clipb and outputs a clip with a one higher bitdepth to avoid the clamping or wraparound issues that would otherwise happen with filters like MakeDiff when forming a difference. This function is usually used together with MergeFullDiff, which can be used to add back the difference.

Unsharp mask:

blur_clip = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
diff_clip = core.std.MakeFullDiff(clip, blur_clip)
sharpened_clip = core.std.MergeFullDiff(clip, diff_clip)
```

### **MaskedMerge**

```
std.MaskedMerge(vnode clipa, vnode clipb, vnode mask[, int[] planes, bint first_plane=0, bint premultiplied=0])

MaskedMerge merges clipa with clipb using the per pixel weights in the mask, where 0 means that clipa is returned unchanged. The mask clip is assumed to be full range for all planes and in the 0-1 interval for float formats regardless of the colorspace. If mask is a grayscale clip or if first_plane is true, the mask’s first plane will be used as the mask for merging all planes. The mask will be bilinearly resized if necessary.

If premultiplied is set the blending is performed as if clipb has been pre-multiplied with alpha. In pre-multiplied mode it is an error to try to merge two frames with mismatched full and limited range since it will most likely cause horrible unintended color shifts. In the other mode it’s just a very, very bad idea.

By default all planes will be processed, but it is also possible to specify a list of the planes to merge in the output. The unprocessed planes will be copied from the first clip.

clipa and clipb must have the same dimensions and format, and the mask must be the same format as the clips or the grayscale equivalent.

How to apply a mask to the first plane:

MaskedMerge(clipa=A, clipb=B, mask=Mask, planes=0)
How to apply the first plane of a mask to the second and third plane:

MaskedMerge(clipa=A, clipb=B, mask=Mask, planes=[1, 2], first_plane=True)
The frame properties are copied from clipa.
```

### **Median**

```
std.Median(vnode clip[, int[] planes=[0, 1, 2]])

Replaces each pixel with the median of the nine pixels in its 3x3 neighbourhood. In other words, the nine pixels are sorted from lowest to highest, and the middle value is picked.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.
```

### **Merge**

```
std.Merge(vnode clipa, vnode clipb[, float[] weight = 0.5])
Merges clipa and clipb using the specified weight for each plane. The default is to use a 0.5 weight for all planes. A zero weight means that clipa is returned unchanged and 1 means that clipb is returned unchanged. If a single weight is specified, it will be used for all planes. If two weights are given then the second value will be used for the third plane as well.

Values outside the 0-1 range are considered to be an error. Specifying more weights than planes in the clips is also an error. The clips must have the same dimensions and format.

How to merge luma:

Merge(clipa=A, clipb=B, weight=[1, 0])
How to merge chroma:

Merge(clipa=A, clipb=B, weight=[0, 1])
The average of two clips:

Merge(clipa=A, clipb=B)
The frame properties are copied from clipa.
```

### **MergeDiff**

```
std.MergeDiff(vnode clipa, vnode clipb[, int[] planes])

Merges back the difference in clipb to clipa and clamps the result. By default all planes are processed. This function is usually used together with MakeDiff, which is normally used to calculate the difference.

Unsharp masking of luma:

blur_clip = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=[0])
diff_clip = core.std.MakeDiff(clip, blur_clip, planes=[0])
sharpened_clip = core.std.MergeDiff(clip, diff_clip, planes=[0])
```

### **MergeFullDiff**

```
std.MergeFullDiff(vnode clipa, vnode clipb)

Merges back the difference in clipb to clipa. Note that the bitdepth of clipb has to be one higher than that of clip. This function is usually used together with MakeFullDiff, which is normally used to calculate the difference.

Unsharp mask:

blur_clip = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
diff_clip = core.std.MakeFullDiff(clip, blur_clip)
sharpened_clip = core.std.MergeFullDiff(clip, diff_clip)
```

### **Minimum/Maximum**

```
std.Minimum(vnode clip[, int[] planes=[0, 1, 2], float threshold, bint[] coordinates=[1, 1, 1, 1, 1, 1, 1, 1]])

Replaces each pixel with the smallest value in its 3x3 neighbourhood. This operation is also known as erosion.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.

threshold
Allows to limit how much pixels are changed. Output pixels will not become less than input - threshold. The default is no limit.

coordinates
Specifies which pixels from the 3x3 neighbourhood are considered. If an element of this array is 0, the corresponding pixel is not considered when finding the minimum value. This must contain exactly 8 numbers.

Here is how each number corresponds to a pixel in the 3x3 neighbourhood:

1 2 3
4   5
6 7 8
```

```
std.Maximum(vnode clip[, int[] planes=[0, 1, 2], float threshold, bint[] coordinates=[1, 1, 1, 1, 1, 1, 1, 1]])

Replaces each pixel with the largest value in its 3x3 neighbourhood. This operation is also known as dilation.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.

threshold
Allows to limit how much pixels are changed. Output pixels will not become less than input - threshold. The default is no limit.

coordinates
Specifies which pixels from the 3x3 neighbourhood are considered. If an element of this array is 0, the corresponding pixel is not considered when finding the maximum value. This must contain exactly 8 numbers.

Here is how each number corresponds to a pixel in the 3x3 neighbourhood:

1 2 3
4   5
6 7 8
```

### **ModifyFrame**

```
std.ModifyFrame(vnode clip, clip[] clips, func selector)

The selector function is called for every single frame and can modify the properties of one of the frames gotten from clips. The additional clips’ properties should only be read and not modified because only one modified frame can be returned.

You must first copy the input frame to make it modifiable. Any frame may be returned as long as it has the same format as the clip. Failure to do so will produce an error. If for conditional reasons you do not need to modify the current frame’s properties, you can simply pass it through. The selector function is passed n, the current frame number, and f, which is a frame or a list of frames if there is more than one clip specified.

If you do not need to modify frame properties but only read them, you should probably be using FrameEval instead.

How to set the property FrameNumber to the current frame number:

def set_frame_number(n, f):
   fout = f.copy()
   fout.props['FrameNumber'] = n
   return fout
...
ModifyFrame(clip=clip, clips=clip, selector=set_frame_number)
How to remove a property:

def remove_property(n, f):
   fout = f.copy()
   del fout.props['FrameNumber']
   return fout
...
ModifyFrame(clip=clip, clips=clip, selector=remove_property)
An example of how to copy certain properties from one clip to another (clip1 and clip2 have the same format):

def transfer_property(n, f):
   fout = f[1].copy()
   fout.props['FrameNumber'] = f[0].props['FrameNumber']
   fout.props['_Combed'] = f[0].props['_Combed']
   return fout
...
ModifyFrame(clip=clip1, clips=[clip1, clip2], selector=transfer_property)
```

### **PEMVerifier**

```
std.PEMVerifier(vnode clip[, float[] upper, float[] lower])
The PEMVerifier is used to check for out-of-bounds pixel values during filter development. It is a public function so badly coded filters won’t go unnoticed.

If no values are set, then upper and lower default to the max and min values allowed in the current format. If an out of bounds value is encountered a frame error is set and the coordinates of the first bad pixel are included in the error message.
```

### **PlaneStats**

```
std.PlaneStats(vnode clipa[, vnode clipb, int plane=0, string prop='PlaneStats'])

This function calculates the min, max and average normalized value of all the pixels in the specified plane and stores the values in the frame properties named propMin, propMax and propAverage.

If clipb is supplied, the absolute normalized difference between the two clips will be stored in propDiff as well.

The normalization means that the average and the diff will always be floats between 0 and 1, no matter what the input format is.
```

### **PreMultiply**

```
std.PreMultiply(vnode clip, vnode alpha)

PreMultiply simply multiplies clip and alpha in order to make it more suitable for later operations. This will yield much better results when resizing and a clip with an alpha channel and MaskedMerge can use it as input. The alpha clip must be the grayscale format equivalent of clip.

Note that limited range pre-multiplied contents excludes the offset. For example with 8 bit input 60 luma and 128 alpha would be calculated as ((60 - 16) * 128)/255 + 16 and not (60 * 128)/255.
```

### **Prewitt/Sobel**

```
std.Prewitt(vnode clip[, int[] planes=[0, 1, 2], float scale=1])
Creates an edge mask using the Prewitt operator.

std.Sobel(vnode clip[, int[] planes=[0, 1, 2], float scale=1])
Creates an edge mask using the Sobel operator.

clip
Clip to process. It must have integer sample type and bit depth between 8 and 16, or float sample type and bit depth of 32. If there are any frames with other formats, an error will be returned.

planes
Specifies which planes will be processed. Any unprocessed planes will be simply copied.

scale
Multiply all pixels by scale before outputting. This can be used to increase or decrease the intensity of edges in the output.
```

### **PropToClip**

```
std.PropToClip(vnode clip[, string prop='_Alpha'])

Extracts a clip from the frames attached to the frame property prop in clip. This function is mainly used to extract a mask/alpha clip that was stored in another one.

It is the inverse of ClipToProp().
```

### **RemoveFrameProps**

```
std.RemoveFrameProps(vnode clip[, string props[]])

Returns clip but with all the frame properties named in props removed. If props is unset them all frame properties are removed.
```

### **Resize**

```
resize.Bilinear(vnode clip[, int width, int height, int format, enum matrix, enum transfer, enum primaries, enum range, enum chromaloc, enum matrix_in, enum transfer_in, enum primaries_in, enum range_in, enum chromaloc_in, float filter_param_a, float filter_param_b, string resample_filter_uv, float filter_param_a_uv, float filter_param_b_uv, string dither_type="none", string cpu_type, float src_left, float src_top, float src_width, float src_height, float nominal_luminance])
resize.Bicubic(vnode clip[, ...])
resize.Point(vnode clip[, ...])
resize.Lanczos(vnode clip[, ...])
resize.Spline16(vnode clip[, ...])
resize.Spline36(vnode clip[, ...])
resize.Spline64(vnode clip[, ...])
resize.Bob(vnode clip, string filter="bicubic", bint tff[, ...])
In VapourSynth the resizers have several functions. In addition to scaling, they also do colorspace conversions and conversions to and from the compat formats. Resize converts a clip of known or unknown format to another clip of known or unknown format, changing only the parameters specified by the user. The resize filters can handle varying size and format input clips and turn them into constant format clips.

If you do not know which resizer to choose, then try Bicubic. It usually makes a good neutral default.

Bob can be used as a rudimentary deinterlacer.

Arguments denoted as type enum may be specified by numerical index (see ITU-T H.265 Annex E.3) or by name. Enums specified by name have their argument name suffixed with “_s”. For example, a destination matrix of BT 709 can be specified either with matrix=1 or with matrix_s="709".

Note that matrix is not an optional argument when converting to YUV. Also note that if no matrix is specified in an input YUV frame’s properties then matrix_in also needs to be set.

The function will return an error if the subsampling restrictions aren’t followed.

If you get an error like:

Resize error 3074: no path between colorspaces (2/2/2 => 1/1/1).
May need to specify additional colorspace parameters.
It usually means the matrix/transfer/primaries are unknown and you have to specify the input colorspace parameters yourself. Note: 2 means “unspecified” according to the ITU-T recommendation.

Resizing is performed per-field for interlaced images, as indicated by the _FieldBased frame property. Source filters may sometimes mark progressive video as interlaced, which can result in sub-optimal resampling quality unless _FieldBased is cleared.

clip:

Accepts all kinds of input.

width, height:

Output image dimensions.

filter:

Scaling method for deinterlacing. See resample_filter_uv for accepted values.

tff:

Field order for deinterlacing. Used when the _FieldBased property is not set.

format:

Output format id.

matrix, transfer, primaries:

Output colorspace specification. If not provided, the corresponding attributes from the input clip will be selected, except for YCoCg and RGB color families, where the corresponding matrix is set by default.

range:

Output pixel range. For integer formats, this allows selection of the legal code values. Even when set, out of range values (BTB/WTW) may be generated. If the input format is of a different color family, the default range is studio/limited for YUV and full-range for RGB.

chromaloc:

Output chroma location. For subsampled formats, specifies the chroma location. If the input format is 4:4:4 or RGB and the output is subsampled, the default location is left-aligned, as per MPEG. Possible chroma locations (ITU-T H.265 Figure E.1): left, center, top_left, top, bottom_left, bottom

matrix_in, transfer_in, primaries_in, range_in, chromaloc_in:

Input colorspace/format specification. If the corresponding frame property is set to a value other than unspecified, the frame property is used instead of this parameter. Default values are set for certain color families. See the equivalent output arguments for more information.

filter_param_a, filter_param_b:

Parameters for the scaler used for RGB and Y-channel. For the bicubic filter, filter_param_a/b represent the “b” and “c” parameters. For the lanczos filter, filter_param_a represents the number of taps.

resample_filter_uv:

Scaling method for UV channels. It defaults to the same as for the Y-channel. The following values can be used with resample_filter_uv: point, bilinear, bicubic, spline16, spline36, lanczos.

filter_param_a_uv, filter_param_b_uv:

Parameters for the scaler used for UV channels.

dither_type:

Dithering method. Dithering is used only for conversions resulting in an integer format. The following dithering methods are available: none, ordered, random, error_diffusion.

cpu_type:

Only used for testing.

src_left, src_top, src_width, src_height:

Used to select the source region of the input to use. Can also be used to shift the image. Defaults to the whole image.

nominal_luminance:

Determines the physical brightness of the value 1.0. The unit is in cd/m^2.

To convert to YV12:

Bicubic(clip=clip, format=vs.YUV420P8, matrix_s="709")
To resize and convert YUV with color information frame properties to planar RGB:

Bicubic(clip=clip, width=1920, height=1080, format=vs.RGB24)
To resize and convert YUV without color information frame properties to planar RGB:

Bicubic(clip=clip, width=1920, height=1080, format=vs.RGB24, matrix_in_s="709")
The following tables list values of selected colorspace enumerations and their abbreviated names. (Numerical value in parentheses.) For all possible values, see ITU-T H.265.

Matrix coefficients (ITU-T H.265 Table E.5):

rgb (0)        Identity
               The identity matrix.
               Typically used for GBR (often referred to as RGB);
               however, may also be used for YZX (often referred to as
               XYZ);
709 (1)        KR = 0.2126; KB = 0.0722
               ITU-R Rec. BT.709-5
unspec (2)     Unspecified
               Image characteristics are unknown or are determined by the
               application.
fcc (4)
470bg (5)      KR = 0.299; KB = 0.114
               ITU-R Rec. BT.470-6 System B, G (historical)
               (functionally the same as the value 6 (170m))
170m (6)       KR = 0.299; KB = 0.114
               SMPTE 170M (2004)
               (functionally the same as the value 5 (470bg))
240m (7)       SMPTE 240M
ycgco (8)      YCgCo
2020ncl (9)    KR = 0.2627; KB = 0.0593
               Rec. ITU-R BT.2020 non-constant luminance system
2020cl (10)    KR = 0.2627; KB = 0.0593
               Rec. ITU-R BT.2020 constant luminance system
chromancl (12) Chromaticity derived non-constant luminance system
chromacl (13)  Chromaticity derived constant luminance system
ictcp (14)     ICtCp
Transfer characteristics (ITU-T H.265 Table E.4):

709 (1)        V = a * Lc0.45 - ( a - 1 ) for 1 >= Lc >= b
               V = 4.500 * Lc for b > Lc >= 0
               Rec. ITU-R BT.709-5
               (functionally the same as the values 6 (601),
               14 (2020_10) and 15 (2020_12))
unspec (2)     Unspecified
               Image characteristics are unknown or are determined by the
               application.
470m (4)       ITU-R Rec. BT.470-6 System M
470bg (5)      ITU-R Rec. BT.470-6 System B, G (historical)
601 (6)        V = a * Lc0.45 - ( a - 1 ) for 1 >= Lc >= b
               V = 4.500 * Lc for b > Lc >= 0
               Rec. ITU-R BT.601-6 525 or 625
               (functionally the same as the values 1 (709),
               14 (2020_10) and 15 (2020_12))
240m (7)       SMPTE 240M
linear (8)     V = Lc for all values of Lc
               Linear transfer characteristics
log100 (9)     Log 1:100 contrast
log316 (10)    Log 1:316 contrast
xvycc (11)     IEC 61966-2-4
srgb (13)      IEC 61966-2-1
2020_10 (14)   V = a * Lc0.45 - ( a - 1 ) for 1 >= Lc >= b
               V = 4.500 * Lc for b > Lc >= 0
               Rec. ITU-R BT.2020
               (functionally the same as the values 1 (709),
               6 (601) and 15 (2020_12))
2020_12 (15)   V = a * Lc0.45 - ( a - 1 ) for 1 >= Lc >= b
               V = 4.500 * Lc for b > Lc >= 0
               Rec. ITU-R BT.2020
               (functionally the same as the values 1 (709),
               6 (601) and 14 (2020_10))
st2084 (16)    SMPTE ST 2084
std-b67 (18)   ARIB std-b67
Color primaries (ITU-T H.265 Table E.3):

709 (1)        primary x y
               green 0.300 0.600
               blue 0.150 0.060
               red 0.640 0.330
               white D65 0.3127 0.3290
               Rec. ITU-R BT.709-5
unspec (2)     Unspecified
               Image characteristics are unknown or are determined by the
               application.
470m (4)       ITU-R Rec. BT.470-6 System M
470bg (5)      ITU-R Rec. BT.470-6 System B, G (historical)
170m (6)       primary x y
               green 0.310 0.595
               blue 0.155 0.070
               red 0.630 0.340
               white D65 0.3127 0.3290
               SMPTE 170M (2004)
               (functionally the same as the value 7 (240m))
240m (7)       primary x y
               green 0.310 0.595
               blue 0.155 0.070
               red 0.630 0.340
               white D65 0.3127 0.3290
               SMPTE 240M (1999)
               (functionally the same as the value 6 (170m))
film (8)
2020 (9)       primary x y
               green 0.170 0.797
               blue 0.131 0.046
               red 0.708 0.292
               white D65 0.3127 0.3290
               Rec. ITU-R BT.2020
st428 (10)     Commonly known as xyz
xyz (10)       Alias for st428
st431-2 (11)   DCI-P3 with traditional white point
st432-1 (12)   DCI-P3
jedec-p22 (22) E.B.U. STANDARD FOR CHROMATICITY TOLERANCES FOR STUDIO MONITORS (3213-E)
               Also known as JEDEC P22
Pixel range (ITU-T H.265 Eq E-4 to E-15):

limited (0) Studio (TV) legal range, 16-235 in 8 bits.
            Y = Clip1Y( Round( ( 1 << ( BitDepthY - 8 ) ) *
                                      ( 219 * E'Y + 16 ) ) )
            Cb = Clip1C( Round( ( 1 << ( BitDepthC - 8 ) ) *
                                       ( 224 * E'PB + 128 ) ) )
            Cr = Clip1C( Round( ( 1 << ( BitDepthC - 8 ) ) *
                                       ( 224 * E'PR + 128 ) ) )

            R = Clip1Y( ( 1 << ( BitDepthY - 8 ) ) *
                               ( 219 * E'R + 16 ) )
            G = Clip1Y( ( 1 << ( BitDepthY - 8 ) ) *
                               ( 219 * E'G + 16 ) )
            B = Clip1Y( ( 1 << ( BitDepthY - 8 ) ) *
                               ( 219 * E'B + 16 ) )
full (1)    Full (PC) dynamic range, 0-255 in 8 bits.
            Y = Clip1Y( Round( ( ( 1 << BitDepthY ) - 1 ) * E'Y ) )
            Cb = Clip1C( Round( ( ( 1 << BitDepthC ) - 1 ) * E'PB +
                                  ( 1 << ( BitDepthC - 1 ) ) ) )
            Cr = Clip1C( Round( ( ( 1 << BitDepthC ) - 1 ) * E'PR +
                                  ( 1 << ( BitDepthC - 1 ) ) ) )

            R = Clip1Y( ( ( 1 << BitDepthY ) - 1 ) * E'R )
            G = Clip1Y( ( ( 1 << BitDepthY ) - 1 ) * E'G )
            B = Clip1Y( ( ( 1 << BitDepthY ) - 1 ) * E'B )
```

### **Reverse**

```
std.Reverse(vnode clip)

Returns a clip with the frame or sample order reversed. For example, a clip with 3 frames would have the frame order 2, 1, 0.

In Python, std.Reverse can also be invoked by slicing a clip.
```

### **SelectEvery**

```
std.SelectEvery(vnode clip, int cycle, int[] offsets[, bint modify_duration=True])

Returns a clip with only some of the frames in every cycle selected. The offsets given must be between 0 and cycle - 1.

Below are some examples of useful operations.

Return even numbered frames, starting with 0:

SelectEvery(clip=clip, cycle=2, offsets=0)
Return odd numbered frames, starting with 1:

SelectEvery(clip=clip, cycle=2, offsets=1)
Fixed pattern 1 in 5 decimation, first frame in every cycle removed:

SelectEvery(clip=clip, cycle=5, offsets=[1, 2, 3, 4])
Duplicate every fourth frame:

SelectEvery(clip=clip, cycle=4, offsets=[0, 1, 2, 3, 3])
In Python, std.SelectEvery can also be invoked by slicing a clip.

If modify_duration is set the clip’s frame rate is multiplied by the number of offsets and divided by cycle. The frame durations are adjusted in the same manner.
```

### **SeparateFields**

```
std.SeparateFields(vnode clip[, bint tff, bint modify_duration=True])
Returns a clip with the fields separated and interleaved.

The tff argument only has an effect when the field order isn’t set for a frame. Setting tff to true means top field first and false means bottom field first.

If modify_duration is set then the output clip’s frame rate is double that of the input clip. The frame durations will also be halved.

The _FieldBased frame property is deleted. The _Field frame property is added.

If no field order is specified in _FieldBased or tff an error will be returned.
```

### **SetFieldBased**

```
std.SetFieldBased(vnode clip, int value)
This is a convenience function. See SetFrameProps if you want to set other properties.

SetFieldBased sets _FieldBased to value and deletes the _Field frame property. The possible values are:

0 = Frame Based

1 = Bottom Field First

2 = Top Field First

For example, if you have source material that’s progressive but has been encoded as interlaced you can set it to be treated as frame based (not interlaced) to improve resizing quality:

clip = core.ffms2.Source("rule6.mkv")
clip = core.std.SetFieldBased(clip, 0)
clip = clip.resize.Bilinear(clip, width=320, height=240)
```

### **SetFrameProp**

```
std.SetFrameProp(vnode clip, string prop[, int[] intval, float[] floatval, string[] data])
Adds a frame property to every frame in clip.

If there is already a property with the name prop in the frames, it will be overwritten.

The type of the property added depends on which of the intval, floatval, or data parameters is used.

The data parameter can only be used to add NULL-terminated strings, not arbitrary binary data.

For example, to set the field order to top field first:

clip = c.std.SetFrameProp(clip, prop="_FieldBased", intval=2)
```

### **SetFrameProps**

```
std.SetFrameProps(vnode clip, ...)

Adds the specified values as a frame property of every frame in clip. If a frame property with the same key already exists it will be replaced.

For example, to set the field order to top field first:

clip = c.std.SetFrameProps(clip, _FieldBased=2)
```

### **SetVideoCache**

```
std.SetVideoCache(vnode clip[, int mode, int fixedsize, int maxsize, int historysize])

Every filter node has a cache associated with it that may or may not be enabled depending on the dependencies and request patterns. This function allows all automatic behavior to be overridden.

The mode option has 3 possible options where 0 always disables caching, 1 always enables the cache and -1 uses the automatically calculated settings. Note that setting mode to -1 will reset the other values to their defaults as well.

The other options are fairly self-explanatory where setting fixedsize prevents the cache from over time altering its maxsize based on request history. The final historysize argument controls how many previous and no longer cached requests should be considered when adjusting maxsize, generally this value should not be touched at all.

Note that setting mode will reset all other options to their defaults.
```

### **ShufflePlanes**

```
std.ShufflePlanes(vnode[] clips, int[] planes, int colorfamily)
ShufflePlanes can extract and combine planes from different clips in the most general way possible. This is both good and bad, as there are almost no error checks.

Most of the returned clip’s properties are implicitly determined from the first clip given to clips.

The clips parameter takes between one and three clips for color families with three planes. In this case clips=[A] is equivalent to clips=[A, A, A] and clips=[A, B] is equivalent to clips=[A, B, B]. For the GRAY color family, which has one plane, it takes exactly one clip.

The argument planes controls which of the input clips’ planes to use. Zero indexed. The first number refers to the first input clip, the second number to the second clip, the third number to the third clip.

The only thing that needs to be specified is colorfamily, which controls which color family (YUV, RGB, GRAY) the output clip will be. Properties such as subsampling are determined from the relative size of the given planes to combine.

ShufflePlanes accepts clips with variable format and dimensions only when extracting a single plane.

Below are some examples of useful operations.

Extract plane with index X. X=0 will mean luma in a YUV clip and R in an RGB clip. Likewise 1 will return the U and G channels, respectively:

ShufflePlanes(clips=clip, planes=X, colorfamily=vs.GRAY)
Swap U and V in a YUV clip:

ShufflePlanes(clips=clip, planes=[0, 2, 1], colorfamily=vs.YUV)
Merge 3 grayscale clips into a YUV clip:

ShufflePlanes(clips=[Yclip, Uclip, Vclip], planes=[0, 0, 0], colorfamily=vs.YUV)
Cast a YUV clip to RGB:

ShufflePlanes(clips=[YUVclip], planes=[0, 1, 2], colorfamily=vs.RGB)
```

### **Splice**

```
std.Splice(vnode[] clips[, bint mismatch=0])

Returns a clip with all clips appended in the given order.

Splicing clips with different formats or dimensions is considered an error unless mismatch is true.

In Python, std.Splice can also be invoked using the addition operator.
```

### **SplitPlanes**

```
std.SplitPlanes(vnode clip)

SplitPlanes returns each plane of the input as separate clips.
```

### **StackVertical/StackHorizontal**

```
std.StackVertical(vnode[] clips)

std.StackHorizontal(vnode[] clips)

Stacks all given clips together. The same format is a requirement. For StackVertical all clips also need to be the same width and for StackHorizontal all clips need to be the same height.

The frame properties are copied from the first clip.
```

### **Transpose**

```
std.Transpose(vnode clip)

Flips the contents of the frames in the same way as a matrix transpose would do. Combine it with FlipVertical or FlipHorizontal to synthesize a left or right rotation. Calling Transpose twice in a row is the same as doing nothing (but slower).

Here is a picture to illustrate what Transpose does:

                          0   5  55
 0   1   1   2   3        1   8  89
 5   8  13  21  34   =>   1  13 144
55  89 144 233 377        2  21 233
                          3  34 377
```

### **Trim**

```
std.Trim(vnode clip[, int first=0, int last, int length])

Trim returns a clip with only the frames between the arguments first and last, or a clip of length frames, starting at first. Trim is inclusive so Trim(clip, first=3, last=3) will return one frame. If neither last nor length is specified, no frames are removed from the end of the clip.

Specifying both last and length is considered to be an error. Likewise is calling Trim in a way that returns no frames, as 0 frame clips are not allowed in VapourSynth.

In Python, std.Trim can also be invoked by slicing a clip.
```

### **Turn180**

```
std.Turn180(vnode clip)

Turns the frames in a clip 180 degrees (to the left, not to the right).
```

## 三,字幕流函数

### **ClipInfo**

```
Prints information about the clip, such as the format and framerate.

This is a convenience function for Text.
```

### **CoreInfo**

```
text.CoreInfo([vnode clip=std.BlankClip(), int alignment=7, int scale=1])

Prints information about the VapourSynth core, such as version and memory use. If no clip is supplied, a default blank one is used.

This is a convenience function for Text.
```

### **FrameNum**

```
text.FrameNum(vnode clip[, int alignment=7, int scale=1])

Prints the current frame number.

This is a convenience function for Text.
```

### **FrameProps**

```
text.FrameProps(vnode clip[, string props=[], int alignment=7, int scale=1])

Prints all properties attached to the frames, or if the props array is given only those properties.

This is a convenience function for Text.
```

### **Text**

```
text.Text(vnode clip, string text[, int alignment=7, int scale=1])

Text is a simple text printing filter. It doesn’t use any external libraries for drawing the text. It uses a built-in bitmap font: the not-bold, 8×16 version of Terminus. The font was not modified, only converted from PCF to an array of bytes.

The font covers Windows-1252, which is a superset of ISO-8859-1 (aka latin1). Unprintable characters get turned into underscores. Long lines get wrapped in a dumb way. Lines that end up too low to fit in the frame are silently dropped.

The alignment parameter takes a number from 1 to 9, corresponding to the positions of the keys on a numpad.

The scale parameter sets an integer scaling factor for the bitmap font.

ClipInfo, CoreInfo, FrameNum, and FrameProps are convenience functions based on Text.
```

## 四,音频流

### **AssumeSampleRate**

```
std.AssumeSampleRate(anode clip[, anode src, int samplerate])
```

### **AudioGain**

```
std.AudioGain(anode clip, float[] gain)

AudioGain can either change the volume of individual channels if a separate gain for each channel is given or if only a single gain value is supplied it’s applied to all channels.

Negative gain values are allowed. Applying a too large gain will lead to clipping in integer formats.
```

### **AudioLoop**

```
std.AudioLoop(anode clip[, int times=0])

Returns a clip with the frames or samples repeated over and over again. If times is less than 1 the clip will be repeated until the maximum clip length is reached, otherwise it will be repeated times times.

In Python, std.AudioLoop can also be invoked using the multiplication operator.
```

### **AudioMix**

```
std.AudioMix(anode[] clips, float[] matrix, int[] channels_out)
AudioMix can mix and combine channels from different clips in the most general way possible.

Most of the returned clip’s properties are implicitly determined from the first clip given to clips.

The clips parameter takes one or more clips with the same format. If the clips are different lengths they’ll be zero extended to that of the longest.

The argument matrix applies the coefficients to each channel of each input clip where the channels are in the numerical order of their channel constants. For example a stereo clip will have its channels presented in the order FRONT_LEFT and then FRONT_RIGHT.

Output channels and order is determined by the channels_out array between input index and output channel happens on the order of lowest output channel identifier to the highest.

Below are some examples of useful operations.

Downmix stereo audio to mono:

AudioMix(clips=clip, matrix=[0.5, 0.5], channels_out=[vs.FRONT_CENTER])
Downmix 5.1 audio:

AudioMix(clips=clip, matrix=[1, 0, 0.7071, 0, 0.7071, 0, 0, 1, 0.7071, 0, 0, 0.7071], channels_out=[vs.FRONT_LEFT, vs.FRONT_RIGHT])
Copy stereo audio to 5.1 and zero the other channels:

AudioMix(clips=c, matrix=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], channels_out=[vs.FRONT_LEFT, vs.FRONT_RIGHT, vs.FRONT_CENTER, vs.LOW_FREQUENCY, vs.BACK_LEFT, vs.BACK_RIGHT])
```

### **AudioReverse**

```
std.AudioReverse(anode clip)

Returns a clip with the frame or sample order reversed. For example, a clip with 3 frames would have the frame order 2, 1, 0.

In Python, std.AudioReverse can also be invoked by slicing a clip.
```

### **AudioSplice**

```
std.AudioSplice(anode[] clips)

Returns a clip with all clips appended in the given order.

Splicing clips with different formats or dimensions is considered an error unless mismatch is true.

In Python, std.AudioSplice can also be invoked using the addition operator.
```

### **AudioTrim**

```
std.AudioTrim(anode clip[, int first=0, int last, int length])

AudioTrim performs exactly the same operation on audio clips but the unit is obviously samples instead of frames.

In Python, std.AudioTrim can also be invoked by slicing a clip.
```

### **BlankAudio**

```
std.BlankAudio([anode clip, int[] channels=[FRONT_LEFT, FRONT_RIGHT], int bits=16, int sampletype=INTEGER, int samplerate=44100, int length=(10*samplerate), bint keep=0])

Generates a new empty clip. This can be useful to have when editing audio or for testing. The default is a 10 second long 44.1kHz 16 bit stereo clip. Instead of specifying every property individually, BlankAudio can also copy the properties from clip. If both an argument such as sampletype, and clip are set, then sampletype will take precedence.

The channels argument is a list of channel constants. Specifying the same channel twice is not allowed.

The possible sampletype values are currently INTEGER (0) and FLOAT (1).

If keep is set, a reference to the same frame is returned on every request. Otherwise a new frame is generated every time. There should usually be no reason to change this setting.
```

### **SetAudioCache**

```
std.SetAudioCache(anode clip[, int mode, int fixedsize, int maxsize, int historysize])

see SetVideoCache
```

### **ShuffleChannels**

```
std.ShuffleChannels(anode[] clips, int[] channels_in, int[] channels_out)

ShuffleChannels can extract and combine channels from different clips in the most general way possible.

Most of the returned clip’s properties are implicitly determined from the first clip given to clips.

The clips parameter takes one or more clips with the same format. If the clips are different lengths they’ll be zero extended to that of the longest.

The argument channels_in controls which of the input clips’ channels to use and takes a channel constants as its argument. Specifying a non-existent channel is an error. If more channels_in than clips values are specified then the last clip in the clips list is reused as a source. In addition to the channel constant it’s also possible to specify the nth channel by using negative numbers.

The output channel mapping is determined by channels_out and corresponds to the input channel order. The number of channels_out entries must be the same as the number of channels_in entries. Specifying the same output channel twice is an error.

Below are some examples of useful operations.

Extract the left channel (assuming it exists):

ShuffleChannels(clips=clip, channels_in=vs.FRONT_LEFT, channels_out=vs.FRONT_LEFT)
Swap left and right audio channels in a stereo clip:

ShuffleChannels(clips=clip, channels_in=[vs.FRONT_RIGHT, vs.FRONT_LEFT], channels_out=[vs.FRONT_LEFT, vs.FRONT_RIGHT])
Swap left and right audio channels in a stereo clip (alternate ordering of arguments):

ShuffleChannels(clips=clip, channels_in=[vs.FRONT_LEFT, vs.FRONT_RIGHT], channels_out=[vs.FRONT_RIGHT, vs.FRONT_LEFT])
Swap left and right audio channels in a stereo clip (alternate indexing):

ShuffleChannels(clips=clip, channels_in=[-2, -1], channels_out=[vs.FRONT_LEFT, vs.FRONT_RIGHT])
Merge 2 mono audio clips into a single stereo clip:

ShuffleChannels(clips=[clipa, clipb], channels_in=[vs.FRONT_LEFT, vs.FRONT_LEFT], channels_out=[vs.FRONT_LEFT, vs.FRONT_RIGHT])
```

### **SplitChannels**

```
std.SplitChannels(anode clip)

SplitChannels returns each audio channel of the input as a separate clip.
```

## 四，输出

### VSPipe

```
vspipe <script> <outfile> [options]

vspipe’s main purpose is to evaluate VapourSynth scripts and output the frames to a file.

If outfile is a hyphen (-), vspipe will write to the standard output.

If outfile is a dot (.), vspipe will do everything as usual, except it will not write the video frames anywhere.
```

### **Options**

```
Options
-a, --arg key=value
Argument to pass to the script environment, it a key with this name and value (str typed) will be set in the globals dict

-s, --start N
Set output frame range (first frame)

-e, --end N
Set output frame range (last frame)

-o, --outputindex N
Select output index

-r, --requests N
Set number of concurrent frame requests

-c, --container <y4m/wav/w64>
Add headers for the specified format to the output

-t, --timecodes FILE
Write timecodes v2 file

-p, --progress
Print progress to stderr

--filter-time
Records the time spent in each filter and prints it out at the end of processing.

-i, --info
Show video info and exit

-g, --graph <simple/full>
Print output node filter graph in dot format to outfile and exit

-v, --version
Show version info and exit
```

### **Examples**

```
Show script info:
vspipe --info script.vpy -

Write to stdout:
vspipe [options] script.vpy -

Request all frames but don’t output them:
vspipe [options] script.vpy .

Write frames 5-100 to file:
vspipe --start 5 --end 100 script.vpy output.raw

Pipe to x264 and write timecodes file:
vspipe script.vpy - --y4m --timecodes timecodes.txt | x264 --demuxer y4m -o script.mkv -

Pass values to a script:
vspipe --arg deinterlace=yes --arg "message=fluffy kittens" script.vpy output.raw
```

