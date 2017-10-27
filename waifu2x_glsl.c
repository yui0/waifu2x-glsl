//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2017 Yuichiro Nakada
//---------------------------------------------------------

// clang -Os waifu2x_glsl.c -o waifu2x_glsl `pkg-config --libs --cflags glesv2 egl gbm` -lglfw -lm
#include <stdlib.h>
#include "gpgpu_glsl.h"

#define PARG_IMPLEMENTATION
#include "parg.h"
#define PARSON_IMPLEMENTATION
#include "parson.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

typedef struct {
	int type;	// MLP, CONV, MAXPOOL
	int act;	// activation function type
	int in;	// input channel
	int out;	// output channel
	int size;	// input size (ch * x * y)
	int width;	// input width
	int height;	// input height
	int ksize;	// kernel size
	int stride;
} CatsEye_Layer;

#define numerus		float
typedef struct {
	// number of each layer
	int layers;
	CatsEye_Layer *u;

	// input layers
	numerus *xdata;
	int xsize;
	// output layers [o = f(z)]
	numerus **z, **o, *odata;
	int osize;
	// error value
	numerus **d, *ddata;
	int dsize;
	// weights
	numerus **w, *wdata;
	int *ws, wsize;
	// bias
	numerus **b, *bdata;
	int *bs, bsize;
} CatsEye;

int CatsEye_loadJson(CatsEye *this, char *name)
{
	JSON_Value *root_value = json_parse_file(name);
	if (json_value_get_type(root_value) != JSONArray) return 1;
	JSON_Array *a = json_value_get_array(root_value);

	this->layers = json_array_get_count(a);
	this->u = malloc(sizeof(CatsEye_Layer)*this->layers);
	this->b = malloc(sizeof(numerus*)*this->layers);
	this->bs = malloc(sizeof(int)*this->layers);
	this->w = malloc(sizeof(numerus*)*this->layers);
	this->ws = malloc(sizeof(int)*this->layers);

	this->bsize = 0;
	for (int i=0; i<this->layers; i++) {
		JSON_Object *o = json_array_get_object(a, i);
		this->bs[i] = this->bsize;
		this->bsize += json_object_get_number(o, "nOutputPlane");
	}
	this->bdata = malloc(sizeof(numerus)*this->bsize);
	for (int i=0; i<this->layers; i++) {
		JSON_Object *o = json_array_get_object(a, i);
		JSON_Array *aa = json_object_get_array(o, "bias");
		for (int j=0; j<json_array_get_count(aa); j++) {
			this->bdata[this->bs[i]+j] = json_array_get_number(aa, j);
		}
	}

	this->wsize = 0;
	for (int i=0; i<this->layers; i++) {
		JSON_Object *o = json_array_get_object(a, i);
		this->ws[i] = this->wsize;
		this->wsize += json_object_get_number(o, "nInputPlane")*json_object_get_number(o, "nOutputPlane")
			*json_object_get_number(o, "kW")*json_object_get_number(o, "kH");
	}
	this->wdata = malloc(sizeof(numerus)*this->wsize);
	for (int i=0; i<this->layers; i++) {
		JSON_Object *o = json_array_get_object(a, i);
		JSON_Array *aa = json_object_get_array(o, "weight");
		int kW = json_object_get_number(o, "kW");
		int kH = json_object_get_number(o, "kH");
		int in = json_object_get_number(o, "nInputPlane");
		int out = json_object_get_number(o, "nOutputPlane");
		this->u[i].ksize = kW;
		this->u[i].in = in;
		this->u[i].out = out;

		for (int j=0; j<out; j++) {
			for (int k=0; k<in; k++) {
				JSON_Array *aaa = json_array_get_array(json_array_get_array(aa, j), k);

				for (int m=0; m<kH; m++) {
					JSON_Array *aaaa = json_array_get_array(aaa, m);
					for (int n=0; n<kW; n++) {
						this->wdata[this->ws[i] +(j*in+k)*kW*kH +m*kW +n] = json_array_get_number(aaaa, n);
					}
				}
			}
		}
	}
	printf("wsize:%d\n", this->wsize);

	json_value_free(root_value);
	return 0;
}

#define XSIZE		256
#define YSIZE		256
#define DATA_XSIZE	4096
#define DATA_YSIZE	2048
#define KERNEL_W	256
#define KERNEL_H	281	// 287136/4/256

char convolution[] = STRINGIFY(

#ifdef GL_ES
precision highp float;
#endif

#define xSize		1./DATA_XSIZE.
#define ySize		1./DATA_YSIZE.

uniform int INPUTPLANE;	// INPUTPLANE/4
uniform vec2 inputOffset[128/4];
uniform vec2 uvpos;
uniform float wpos;

uniform sampler2D X;
uniform sampler2D W;
uniform vec4 bias[128/4];	// 4

varying vec2 uv;

// https://qiita.com/YVT/items/c695ab4b3cf7faa93885
// x:GL_WRAP, y:GL_NEAREST
// arg = vec2(1./size.x, 1./size.x/size.y);
/*vec4 fetchElement(sampler2D tex, float index, vec2 arg)
{
	return texture2D(tex, arg * (index+0.5));
}*/

void main()
{
	// calc uv pos [0-1, 0-1]
//	vec2 a = uv*vec2(XSIZE./DATA_XSIZE., YSIZE./DATA_YSIZE.);
	vec2 a = uv*uvpos;
	vec2 oplane = floor(a/vec2(XSIZE./DATA_XSIZE., YSIZE./DATA_YSIZE.));	// /0.0625 (256)
	a -= oplane * vec2(XSIZE./DATA_XSIZE., YSIZE./DATA_YSIZE.);
	int op = int(oplane.x + oplane.y*16.);

	// calc w pos
	const vec2 arg = vec2(1./KERNEL_W., 1./KERNEL_W./KERNEL_H.);
	vec2 pos[4];
	pos[0] = arg * (wpos+ float(op* INPUTPLANE *9) +0.5);	// arg * (index+0.5)
	vec2 n = arg * float(INPUTPLANE *9);
	pos[1] = pos[0] + n;
	pos[2] = pos[1] + n;
	pos[3] = pos[2] + n;

	vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
	for (int i=0; i<INPUTPLANE; i++) {
		vec2 tuv = a + inputOffset[i];

		vec4 p[9];
		p[0] = texture2D(X, tuv + vec2(-xSize, -ySize));
		p[1] = texture2D(X, tuv + vec2(   0.0, -ySize));
		p[2] = texture2D(X, tuv + vec2( xSize, -ySize));
		p[3] = texture2D(X, tuv + vec2(-xSize,    0.0));
		p[4] = texture2D(X, tuv + vec2(   0.0,    0.0));
		p[5] = texture2D(X, tuv + vec2( xSize,    0.0));
		p[6] = texture2D(X, tuv + vec2(-xSize,  ySize));
		p[7] = texture2D(X, tuv + vec2(   0.0,  ySize));
		p[8] = texture2D(X, tuv + vec2( xSize,  ySize));

		vec4 a[9];
		a[0] = texture2D(W, pos[0]); pos[0] += arg;	// 1-4
		a[1] = texture2D(W, pos[0]); pos[0] += arg;	// 5-8
		a[2] = texture2D(W, pos[0]); pos[0] += arg;	// 9-12
		a[3] = texture2D(W, pos[0]); pos[0] += arg;	// 13-16
		a[4] = texture2D(W, pos[0]); pos[0] += arg;	// 17-20
		a[5] = texture2D(W, pos[0]); pos[0] += arg;	// 21-24
		a[6] = texture2D(W, pos[0]); pos[0] += arg;	// 25-28
		a[7] = texture2D(W, pos[0]); pos[0] += arg;	// 29-32
		a[8] = texture2D(W, pos[0]); pos[0] += arg;	// 33-36

		sum.x += dot(vec3(p[0].x, p[1].x, p[2].x), a[0].xyz);
		sum.x += dot(vec3(p[3].x, p[4].x, p[5].x), vec3(a[0].w, a[1].x, a[1].y));
		sum.x += dot(vec3(p[6].x, p[7].x, p[8].x), vec3(a[1].z, a[1].w, a[2].x));

		sum.x += dot(vec3(p[0].y, p[1].y, p[2].y), a[2].yzw);
		sum.x += dot(vec3(p[3].y, p[4].y, p[5].y), a[3].xyz);
		sum.x += dot(vec3(p[6].y, p[7].y, p[8].y), vec3(a[3].w, a[4].x, a[4].y));

		sum.x += dot(vec3(p[0].z, p[1].z, p[2].z), vec3(a[4].z, a[4].w, a[5].x));
		sum.x += dot(vec3(p[3].z, p[4].z, p[5].z), a[5].yzw);
		sum.x += dot(vec3(p[6].z, p[7].z, p[8].z), a[6].xyz);

		sum.x += dot(vec3(p[0].w, p[1].w, p[2].w), vec3(a[6].w, a[7].x, a[7].y));
		sum.x += dot(vec3(p[3].w, p[4].w, p[5].w), vec3(a[7].z, a[7].w, a[8].x));
		sum.x += dot(vec3(p[6].w, p[7].w, p[8].w), a[8].yzw);

		a[0] = texture2D(W, pos[1]); pos[1] += arg;	// 1-4
		a[1] = texture2D(W, pos[1]); pos[1] += arg;	// 5-8
		a[2] = texture2D(W, pos[1]); pos[1] += arg;	// 9-12
		a[3] = texture2D(W, pos[1]); pos[1] += arg;	// 13-16
		a[4] = texture2D(W, pos[1]); pos[1] += arg;	// 17-20
		a[5] = texture2D(W, pos[1]); pos[1] += arg;	// 21-24
		a[6] = texture2D(W, pos[1]); pos[1] += arg;	// 25-28
		a[7] = texture2D(W, pos[1]); pos[1] += arg;	// 29-32
		a[8] = texture2D(W, pos[1]); pos[1] += arg;	// 33-36

		sum.y += dot(vec3(p[0].x, p[1].x, p[2].x), a[0].xyz);
		sum.y += dot(vec3(p[3].x, p[4].x, p[5].x), vec3(a[0].w, a[1].x, a[1].y));
		sum.y += dot(vec3(p[6].x, p[7].x, p[8].x), vec3(a[1].z, a[1].w, a[2].x));

		sum.y += dot(vec3(p[0].y, p[1].y, p[2].y), a[2].yzw);
		sum.y += dot(vec3(p[3].y, p[4].y, p[5].y), a[3].xyz);
		sum.y += dot(vec3(p[6].y, p[7].y, p[8].y), vec3(a[3].w, a[4].x, a[4].y));

		sum.y += dot(vec3(p[0].z, p[1].z, p[2].z), vec3(a[4].z, a[4].w, a[5].x));
		sum.y += dot(vec3(p[3].z, p[4].z, p[5].z), a[5].yzw);
		sum.y += dot(vec3(p[6].z, p[7].z, p[8].z), a[6].xyz);

		sum.y += dot(vec3(p[0].w, p[1].w, p[2].w), vec3(a[6].w, a[7].x, a[7].y));
		sum.y += dot(vec3(p[3].w, p[4].w, p[5].w), vec3(a[7].z, a[7].w, a[8].x));
		sum.y += dot(vec3(p[6].w, p[7].w, p[8].w), a[8].yzw);

		a[0] = texture2D(W, pos[2]); pos[2] += arg;	// 1-4
		a[1] = texture2D(W, pos[2]); pos[2] += arg;	// 5-8
		a[2] = texture2D(W, pos[2]); pos[2] += arg;	// 9-12
		a[3] = texture2D(W, pos[2]); pos[2] += arg;	// 13-16
		a[4] = texture2D(W, pos[2]); pos[2] += arg;	// 17-20
		a[5] = texture2D(W, pos[2]); pos[2] += arg;	// 21-24
		a[6] = texture2D(W, pos[2]); pos[2] += arg;	// 25-28
		a[7] = texture2D(W, pos[2]); pos[2] += arg;	// 29-32
		a[8] = texture2D(W, pos[2]); pos[2] += arg;	// 33-36

		sum.z += dot(vec3(p[0].x, p[1].x, p[2].x), a[0].xyz);
		sum.z += dot(vec3(p[3].x, p[4].x, p[5].x), vec3(a[0].w, a[1].x, a[1].y));
		sum.z += dot(vec3(p[6].x, p[7].x, p[8].x), vec3(a[1].z, a[1].w, a[2].x));

		sum.z += dot(vec3(p[0].y, p[1].y, p[2].y), a[2].yzw);
		sum.z += dot(vec3(p[3].y, p[4].y, p[5].y), a[3].xyz);
		sum.z += dot(vec3(p[6].y, p[7].y, p[8].y), vec3(a[3].w, a[4].x, a[4].y));

		sum.z += dot(vec3(p[0].z, p[1].z, p[2].z), vec3(a[4].z, a[4].w, a[5].x));
		sum.z += dot(vec3(p[3].z, p[4].z, p[5].z), a[5].yzw);
		sum.z += dot(vec3(p[6].z, p[7].z, p[8].z), a[6].xyz);

		sum.z += dot(vec3(p[0].w, p[1].w, p[2].w), vec3(a[6].w, a[7].x, a[7].y));
		sum.z += dot(vec3(p[3].w, p[4].w, p[5].w), vec3(a[7].z, a[7].w, a[8].x));
		sum.z += dot(vec3(p[6].w, p[7].w, p[8].w), a[8].yzw);

		a[0] = texture2D(W, pos[3]); pos[3] += arg;	// 1-4
		a[1] = texture2D(W, pos[3]); pos[3] += arg;	// 5-8
		a[2] = texture2D(W, pos[3]); pos[3] += arg;	// 9-12
		a[3] = texture2D(W, pos[3]); pos[3] += arg;	// 13-16
		a[4] = texture2D(W, pos[3]); pos[3] += arg;	// 17-20
		a[5] = texture2D(W, pos[3]); pos[3] += arg;	// 21-24
		a[6] = texture2D(W, pos[3]); pos[3] += arg;	// 25-28
		a[7] = texture2D(W, pos[3]); pos[3] += arg;	// 29-32
		a[8] = texture2D(W, pos[3]); pos[3] += arg;	// 33-36

		sum.w += dot(vec3(p[0].x, p[1].x, p[2].x), a[0].xyz);
		sum.w += dot(vec3(p[3].x, p[4].x, p[5].x), vec3(a[0].w, a[1].x, a[1].y));
		sum.w += dot(vec3(p[6].x, p[7].x, p[8].x), vec3(a[1].z, a[1].w, a[2].x));

		sum.w += dot(vec3(p[0].y, p[1].y, p[2].y), a[2].yzw);
		sum.w += dot(vec3(p[3].y, p[4].y, p[5].y), a[3].xyz);
		sum.w += dot(vec3(p[6].y, p[7].y, p[8].y), vec3(a[3].w, a[4].x, a[4].y));

		sum.w += dot(vec3(p[0].z, p[1].z, p[2].z), vec3(a[4].z, a[4].w, a[5].x));
		sum.w += dot(vec3(p[3].z, p[4].z, p[5].z), a[5].yzw);
		sum.w += dot(vec3(p[6].z, p[7].z, p[8].z), a[6].xyz);

		sum.w += dot(vec3(p[0].w, p[1].w, p[2].w), vec3(a[6].w, a[7].x, a[7].y));
		sum.w += dot(vec3(p[3].w, p[4].w, p[5].w), vec3(a[7].z, a[7].w, a[8].x));
		sum.w += dot(vec3(p[6].w, p[7].w, p[8].w), a[8].yzw);
	}
	sum += bias[op];
	sum = max(sum, 0.0) + min(sum, 0.0) * 0.1;
	gl_FragColor = sum;
//	gl_FragColor = texture2D(X, a);
}

);

void *recalloc(void *p, int s, int ss)
{
	void *r = calloc(ss, 1);
	if (!r) return 0;
	memcpy(r, p, s);
	free(p);
	return r;
}

void result(char *name, int w, int h)
{
	float *d = coReadDataf(h, w, 0);
/*	for (int i=0; i<h; i++) {
		for (int j=0; j<w; j++) printf("%2.2f ", d[(i*w+j)*4]);
		printf("\n");
	}
	printf("\n");*/

	unsigned char *o = calloc(w*h, 1);
	for (int y=0; y<h; y++) {
		for (int x=0; x<w; x++) {
			o[y*w+x] = d[(y*w+x)*4]*255;
		}
	}
	stbi_write_png(name, w, h, 1, o, 0);
	free(o);
	free(d);
}

#define DEBUG
#ifdef DEBUG
#define debug(x)	{x;}
#else
#define debug(x)
#endif

int waifu2x_glsl(char *name, char *model)
{
	unsigned char *pixels;
	int w, h, bpp;
	pixels = stbi_load(name, &w, &h, &bpp, 3);
	printf("%s %dx%d %d\n", name, w, h, bpp);

	unsigned char *pix = malloc(w*2*h*2*bpp);
	stbir_resize_uint8_srgb(pixels, w, h, 0, pix, w*2, h*2, 0, bpp, -1, 0);
	stbi_image_free(pixels);
	debug(stbi_write_jpg("output.jpg", w*2, h*2, bpp, pix, 0));

	unsigned char *p = malloc(256*256*3);
	unsigned char *yuv = calloc(256*256*3/2, 1);
//	unsigned char *y = yuv;
//	unsigned char *u = yuv +256*256;
//	unsigned char *v = yuv +256*256 +((256+1)/2)*((256+1)/2);
	float *f = calloc(256*256*4, sizeof(float));
//	float *u = calloc(256*256, sizeof(float));
//	float *v = calloc(256*256, sizeof(float));
	for (int y=0; y<256; y++) {
		for (int x=0; x<256; x++) {
			unsigned char r = pix[(y*w*2+x)*3];
			unsigned char g = pix[(y*w*2+x)*3+1];
			unsigned char b = pix[(y*w*2+x)*3+2];
			p[(y*256+x)*3] = 0.299*r +0.587*g +0.114*b;
			p[(y*256+x)*3+1] = -0.1687*r -0.3313*g +0.500*b +128;
			p[(y*256+x)*3+2] = 0.500*r -0.4187*g -0.0813*b +128;
			yuv[y*256+x] = 0.299*r +0.587*g +0.114*b;

			f[(y*256+x)*4] = (0.298912*r +0.586611*g +0.114478*b)/255.0;	// CCIR Rec.601
//			u[y*256+x] = -0.147*r -0.289*g +0.436*b;
//			v[y*256+x] = 0.615*r -0.515*g -0.100*b;
		}
	}
	debug(stbi_write_png("output_256.png", 256, 256, 3, p, 0));
	debug(stbi_write_png("output_y.png", 256, 256, 1, yuv, 0));
	free(pix);

	CatsEye cat;
	CatsEye_loadJson(&cat, model);
	cat.wdata = recalloc(cat.wdata, sizeof(numerus)*cat.wsize, sizeof(numerus)*KERNEL_W*KERNEL_H*4);
	cat.bdata = recalloc(cat.bdata, sizeof(numerus)*cat.bsize, sizeof(numerus)*(cat.bsize+3));

	coInit();

	GLuint prog = coCreateProgram(convolution);
	coBindVertices(prog);

	GLuint texture[3];
	texture[0] = coCreateDataTexture(DATA_YSIZE, DATA_XSIZE, 0, GL_FLOAT);
	coTransferData(texture[0], 0, 0, XSIZE, YSIZE, GL_FLOAT, f);
	texture[1] = coCreateDataTexture(DATA_YSIZE, DATA_XSIZE, 0, GL_FLOAT);
	texture[2] = coCreateDataTexture(KERNEL_H, KERNEL_W, cat.wdata, GL_FLOAT);
	coBindInputTexture(prog, texture[2], GL_TEXTURE1, "W");

	float ioffset[128/4*2];
	for (int i=0; i<128/4; i++) {
		ioffset[i*2] = (i % 16) / 16.0;
		ioffset[i*2+1] = floor(i / 16.0) / 8.0;
//		ioffset[i*2] = (i % 16) / 16.0 +0.5/DATA_XSIZE;
//		ioffset[i*2+1] = floor(i / 16.0) / 8.0 +0.5/DATA_XSIZE;
		//printf("%f %f\n", ioffset[i*2], ioffset[i*2+1]);
	}
	coUniform2fv(prog, "inputOffset", 128/4, ioffset);

	int n = 0;
	int r = 1;
	//printf("%d\n", cat.layers);
	for (int i=0; i<cat.layers; i++) {
		int a = (cat.u[i].out+3)/4;
		int w = a>16 ? 16 : a;
		int h = (a+15)/16;
		printf("%d %d %dx%d %d %d\n", cat.u[i].in, cat.u[i].out, w, h, (cat.u[i].in+3)/4, cat.ws[i]);

		coUniform1i(prog, "INPUTPLANE", (cat.u[i].in+3)/4);
		coUniform4fv(prog, "bias", w, &cat.bdata[cat.bs[i]]); coAssert();
		coUniform2f(prog, "uvpos", (float)XSIZE*w/DATA_XSIZE, (float)YSIZE*h/DATA_YSIZE);
		coUniform1f(prog, "wpos", (float)cat.ws[i]/4);
		coBindInputTexture(prog, texture[n], GL_TEXTURE0, "X");
		coBindOutputTexture(YSIZE*h, XSIZE*w, texture[r]);
		coCompute();
		n ^= 1;	// swap
		r ^= 1;
#ifdef DEBUG
		char *buff[256];
		sprintf(buff, "output2x_%02d.png", i+1);
		result(buff, XSIZE*w, YSIZE*h);
#endif
	}

	float *d = coReadDataf(YSIZE, XSIZE, 0);
	unsigned char *o = calloc(XSIZE*YSIZE, 3);
	for (int y=0; y<YSIZE; y++) {
		for (int x=0; x<XSIZE; x++) {
			//int yy = (d[(y*XSIZE+x)*4]+1.0)*255;
			//printf("%2.2f ",d[(y*XSIZE+x)*4]);
			int yy = -196*d[(y*XSIZE+x)*4];
			if (yy<0 || yy>255) printf("%d ",yy);
/*			yy = yy<0 ? 0: yy>255 ? 255: yy;
			o[(y*XSIZE+x)*3] = yy;
			o[(y*XSIZE+x)*3+1] = yy;
			o[(y*XSIZE+x)*3+2] = yy;*/

//			unsigned char yy = 255-d[(y*XSIZE+x)*4]*255;
//			unsigned char yy = p[(y*256+x)*3];
			unsigned char u = p[(y*256+x)*3+1];
			unsigned char v = p[(y*256+x)*3+2];
			o[(y*XSIZE+x)*3] = yy +1.402 *(v-128);
			o[(y*XSIZE+x)*3+1] = yy -0.34414 *(u-128) -0.71414*(v-128);
			o[(y*XSIZE+x)*3+2] = yy +1.772 *(u-128);

/*			float yy = -255*d[(y*XSIZE+x)*4];
			if (yy<0 || yy>255) printf("%2.2f ",yy);
			o[(y*XSIZE+x)*3] = yy +1.140*v[y*256+x];
			o[(y*XSIZE+x)*3+1] = yy -0.395*u[y*256+x] -0.580*v[y*256+x];
			o[(y*XSIZE+x)*3+2] = yy +2.032*u[y*256+x];*/
		}
	}
	stbi_write_png("output2x.png", XSIZE, YSIZE, 3, o, 0);
	free(o);
	free(d);

	free(yuv);
	free(f);
	free(p);

	coTerm();
	return 0;
}

int main(int argc, char* argv[])
{
	char *name;
	char *model = "noise1_model.json";
	struct parg_state ps;
	int c;

	parg_init(&ps);
	while ((c = parg_getopt(&ps, argc, argv, "hm:")) != -1) {
		switch (c) {
		case 1:
			name = (char*)ps.optarg;
			break;
		case 'm':
			model = (char*)ps.optarg;
			break;
		case 'h':
		default:
//			usage(stderr, argc, argv);
			return 1;
		}
	}
	waifu2x_glsl(name, model);

	return 0;
}
