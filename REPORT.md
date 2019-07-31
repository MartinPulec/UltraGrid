# Measuring compiler optimizations
The aim of this short report is to evaluate the influence of different compiler optimizations. As a test scenario, a tweaked UltraGrid was used (provided that this report is in GIT, it can be directly compiled as `bin/uv` (main() function of the UltraGrid was replaced).

*Note:* Useful online tool is [Compiler explorer](https://godbolt.org/). Then
also
[this](https://github.com/opcm/pcm) and
[this](https://github.com/andikleen/pmu-tools/). Also following command can be used:

    perf stat -p `pidof uv`

## Environment
Measured on:

* CPU: Intel(R) Core(TM) i7-5960X CPU @ 3.00GHz
* OS: Ubuntu 18.04
* Kernel: 4.4.0-140-generic
* Kernel opts: BOOT\_IMAGE=/vmlinuz-4.4.0-140-generic root=/dev/mapper/w54--151--vg-root ro panic=60
* Compiler: gcc version 7.4.0 (Ubuntu 7.4.0-1ubuntu1~18.04)

## Conclusion
* -O3 is almost always better then O2 except for vc\_copylineR12L which is
  slightly slower
* -Ofast is also _slower_ than -O3 in general
* it seems like that only few functions are effectively optimized by the
  compiler (O3 vs 02), although increasing the level obviously adds more code
  including SSE instructions

## Measuement
### -O2
    vc_copylineDVS10: 0.0486009
    vc_copylinev210: 0.0600656
    vc_copylineYUYV: 0.0152055
    vc_copyliner10k: 0.0926317
    vc_copylineR12L: 0.0822971
    vc_copylineRGBA: 0.00843614
    vc_copylineDVS10toV210: 0.0623579
    vc_copylineRGBAtoRGB: 0.123642
    vc_copylineABGRtoRGB: 0.12343
    vc_copylineRGBAtoRGBwithShift: 0.123668
    vc_copylineRGBtoRGBA: 0.0657934
    vc_copylineRGBtoUYVY: 0.345439
    vc_copylineRGBtoUYVY_SSE: 0.0767643
    vc_copylineRGBtoGrayscale_SSE: 0.0901018
    vc_copylineRGBtoR12L: 0.0608974
    vc_copylineR12LtoRG48: 0.0535392
    vc_copylineR12LtoRGB: 0.058063
    vc_copylineRG48toR12L: 0.0588878
    vc_copylineRG48toRGBA: 0.0659951
    vc_copylineUYVYtoRGB: 0.190902
    vc_copylineUYVYtoRGB_SSE: 0.0396108
    vc_copylineUYVYtoGrayscale: 0.053362
    vc_copylineYUYVtoRGB: 0.190921
    vc_copylineBGRtoUYVY: 0.31919
    vc_copylineRGBAtoUYVY: 0.239021
    vc_copylineBGRtoRGB: 0.0977738
    vc_copylineDPX10toRGBA: 0.0779931
    TODO: incomplete vc_copylineDPX10toRGB implementation!
    vc_copylineDPX10toRGB: 0.0523912
    vc_copylineRGB: 0.00839912
    
    real	0m3.089s
    user	0m3.004s
    sys	0m0.084s

### -O3
    vc_copylineDVS10: 0.0443529
    vc_copylinev210: 0.0600515
    vc_copylineYUYV: 0.0151641
    vc_copyliner10k: 0.0926405
    vc_copylineR12L: 0.0986299
    vc_copylineRGBA: 0.00842301
    vc_copylineDVS10toV210: 0.0187427
    vc_copylineRGBAtoRGB: 0.0937321
    vc_copylineABGRtoRGB: 0.0936926
    vc_copylineRGBAtoRGBwithShift: 0.0937165
    vc_copylineRGBtoRGBA: 0.0397384
    vc_copylineRGBtoUYVY: 0.345509
    vc_copylineRGBtoUYVY_SSE: 0.0765756
    vc_copylineRGBtoGrayscale_SSE: 0.0901611
    vc_copylineRGBtoR12L: 0.0608953
    vc_copylineR12LtoRG48: 0.0532898
    vc_copylineR12LtoRGB: 0.0562404
    vc_copylineRG48toR12L: 0.0592407
    vc_copylineRG48toRGBA: 0.0660487
    vc_copylineUYVYtoRGB: 0.190929
    vc_copylineUYVYtoRGB_SSE: 0.0397013
    vc_copylineUYVYtoGrayscale: 0.0150938
    vc_copylineYUYVtoRGB: 0.19078
    vc_copylineBGRtoUYVY: 0.31868
    vc_copylineRGBAtoUYVY: 0.126215
    vc_copylineBGRtoRGB: 0.0976314
    vc_copylineDPX10toRGBA: 0.0186347
    TODO: incomplete vc_copylineDPX10toRGB implementation!
    vc_copylineDPX10toRGB: 0.0305984
    vc_copylineRGB: 0.00842741
    
    real	0m2.708s
    user	0m2.612s
    sys	0m0.092s

### -Ofast
    vc_copylineDVS10: 0.0445358
    vc_copylinev210: 0.060222
    vc_copylineYUYV: 0.0149841
    vc_copyliner10k: 0.0930155
    vc_copylineR12L: 0.0989949
    vc_copylineRGBA: 0.00845052
    vc_copylineDVS10toV210: 0.0187319
    vc_copylineRGBAtoRGB: 0.0937489
    vc_copylineABGRtoRGB: 0.0937014
    vc_copylineRGBAtoRGBwithShift: 0.0937393
    vc_copylineRGBtoRGBA: 0.0397626
    vc_copylineRGBtoUYVY: 0.345531
    vc_copylineRGBtoUYVY_SSE: 0.0765072
    ^Cvc_copylineRGBtoGrayscale_SSE: 0.0903569
    vc_copylineRGBtoR12L: 0.0610354
    vc_copylineR12LtoRG48: 0.0534733
    vc_copylineR12LtoRGB: 0.0563192
    vc_copylineRG48toR12L: 0.0588506
    vc_copylineRG48toRGBA: 0.0660932
    vc_copylineUYVYtoRGB: 0.276704
    vc_copylineUYVYtoRGB_SSE: 0.0386543
    vc_copylineUYVYtoGrayscale: 0.0150524
    vc_copylineYUYVtoRGB: 0.276389
    vc_copylineBGRtoUYVY: 0.31869
    vc_copylineRGBAtoUYVY: 0.126382
    vc_copylineBGRtoRGB: 0.0977016
    vc_copylineDPX10toRGBA: 0.018727
    TODO: incomplete vc_copylineDPX10toRGB implementation!
    vc_copylineDPX10toRGB: 0.030596
    vc_copylineRGB: 0.00842203
    
    real	0m2.879s
    user	0m2.764s
    sys	0m0.112s
