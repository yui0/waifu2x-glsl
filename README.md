# waifu2x-glsl

Fast waifu2x converter with GPU optimization.
Using GLSL.

## Platform

- Linux
- macOS
- Windows

## How to build

```bash
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

## References

- https://github.com/yui0/catseye
- [Image Super-Resolution Using Deep Convolutional Networks](http://arxiv.org/abs/1501.00092)
- [EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis](https://arxiv.org/abs/1612.07919)
- Waifu2x
  - Original implementation: https://github.com/nagadomi/waifu2x
  - https://github.com/kioku-systemk/waifu2x_webgl
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
- [OpenGL on Ma]c(http://asa-no-blog.hatenablog.com/entry/2017/08/26/235737)
