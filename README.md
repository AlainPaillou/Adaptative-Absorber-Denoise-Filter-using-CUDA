# Adaptative-Absorber-Denoise-Filter-using-CUDA

This test software is supposed to allow you to test my noise removal function (AADF) based on Python & CUDA.

This filter works only with video because it needs at least 2 consecutives images (N-1 and N) to perform noise removal.

You will need a NVIDIA GPU to run this program.

The good :
This noise removal routine is quite fast (about 30ms per frame for FullHD movie) and give very good results with heavy noise while preserving details and sharpness.

The bad :
This filter works with quite static videos. Motions in the video will create ghosting.

The filter gets one parameter : dynamic response of the filter which can get 2 values :
0 : low dynamic response. Will give better quality but ghosting will be high
1 : high dynamic response. Will give lower quality but ghosting will be low

This filter compare 2 images (N-1 and N). It allow big changes and decrease small changes for each pixel.

I created this filter for astronomy use (wide field deep sky survey).

Licence :
This filter is free to use and modify for personal use only.
This filter is NOT free of use for professional use or/and commercial use.
