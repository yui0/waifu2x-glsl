# waifu2x-glsl

Fast waifu2x converter with GPU optimization.
Using GLSL.

## Platform

- Linux with GLEW
- macOS with GLEW
- Windows with [GLFW](http://www.glfw.org/) and GLEW
  - nupengl.core with NuGet

## How to build on macOS

```
$ #/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ brew install glfw
$ make
```

## How to build on Linux

```bash
# #dnf install mesa-libgbm-devel libdrm-devel mesa-libGL-devel mesa-libEGL-devel mesa-libGLES-devel
# dnf install glfw-devel
$ make
```

## How to use

```bash
$ ./waifu2x_glsl -h
Usage: ./waifu2x_glsl [options] file

Options:
-h                 Print this message
-m <model name>    waifu2x model name [noise2_model.json...]
-s <scale>         Magnification [1.0, 1.6, 2.0...]
-o <output name>   output file name [*.png, *.jpg]

$ ./waifu2x_glsl -s 1.0 nyanko.jpg
$ ./waifu2x_glsl -m vgg_7/art_y/noise3_model.json nyanko.jpg
```

## How to work

![01.Nyanko](nyanko_01.png "01")
![02.Nyanko](nyanko_02.png "02")
![03.Nyanko](nyanko_03.png "03")
![04.Nyanko](nyanko_04.png "04")
![05.Nyanko](nyanko_05.png "05")
![06.Nyanko](nyanko_06.png "06")
![07.Nyanko](nyanko_07.png "07")

## Demo

### Original
![Original](waifu_s.jpg)

### Normal Resize
![Normal](waifu_d.jpg)

### Waifu2x Resize
- ./waifu2x_glsl -s 2.0 -m noise2_model.json waifu_s.jpg -o waifu_d.png
![Waifu2x](waifu_d.png)

## References

- https://github.com/yui0/catseye
- [Image Super-Resolution Using Deep Convolutional Networks](http://arxiv.org/abs/1501.00092)
- [EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis](https://arxiv.org/abs/1612.07919)
- Waifu2x
  - [Waifu2x webgl version](https://github.com/kioku-systemk/waifu2x_webgl) / [GLSL code](https://gist.github.com/yui0/a9a75c93b9e7c6a08f905ed548b4b17c)
  - [Original implementation](https://github.com/nagadomi/waifu2x)
  - https://github.com/ueshita/waifu2x-converter-glsl
  - https://stanko.github.io/super-resolution-image-resizer
  - https://www.slideshare.net/KosukeNakago/seranet
- GLSL
  - https://github.com/transcranial/keras-js/tree/master/src/webgl
  - https://github.com/scienceai/neocortex/blob/master/src/lib/webgl/matmul_fragment_shader.glsl
  - https://tenso.rs/demos/fast-neural-style/
- Picture
  - Nyanko: https://www.illust-box.jp/member/view/7263/
  - Nyanko: http://www.poipoi.com/yakko/cgi-bin/sb/log/eid5173.html#more-5173
- [OpenGL on Mac](http://asa-no-blog.hatenablog.com/entry/2017/08/26/235737)
