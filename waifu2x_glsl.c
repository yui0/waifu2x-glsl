//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2018 Yuichiro Nakada
//---------------------------------------------------------

// clang -Os waifu2x_glsl.c -o waifu2x_glsl `pkg-config --libs --cflags glesv2 egl gbm` -lglfw -lm
#include <stdlib.h>
#include <stdint.h>
#include "gpgpu_glsl.h"
#include "clock.h"

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

#define real		float
typedef struct {
	// number of each layer
	int layers;
	CatsEye_Layer *u;

	// input layers
	real *xdata;
	int xsize;
	// output layers [o = f(z)]
	real **z, **o, *odata;
	int osize;
	// error value
	real **d, *ddata;
	int dsize;
	// weights
	real **w, *wdata;
	int *ws, wsize;
	// bias
	real **b, *bdata;
	int *bs, bsize;
} CatsEye;

int CatsEye_loadJson(CatsEye *this, char *name)
{
	JSON_Value *root_value = json_parse_file(name);
	if (json_value_get_type(root_value) != JSONArray) return 1;
	JSON_Array *a = json_value_get_array(root_value);

	this->layers = json_array_get_count(a);
	this->u = malloc(sizeof(CatsEye_Layer)*this->layers);
	this->b = malloc(sizeof(real*)*this->layers);
	this->bs = malloc(sizeof(int)*this->layers);
	this->w = malloc(sizeof(real*)*this->layers);
	this->ws = malloc(sizeof(int)*this->layers);

	this->bsize = 0;
	for (int i=0; i<this->layers; i++) {
		JSON_Object *o = json_array_get_object(a, i);
		this->bs[i] = this->bsize;
		this->bsize += json_object_get_number(o, "nOutputPlane");
	}
	this->bdata = malloc(sizeof(real)*this->bsize);
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
	this->wdata = malloc(sizeof(real)*this->wsize);
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

uniform int INPUTPLANE;	// /4
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
	vec2 a = uv*uvpos;
//	vec2 a = uv*uvpos -vec2(xSize/2., ySize/2.);
//	vec2 a = (uv+vec2(xSize/2., ySize/2.))*uvpos;
	vec2 oplane = floor(a/vec2(XSIZE./DATA_XSIZE., YSIZE./DATA_YSIZE.));	// /0.0625 (256)
	a -= oplane * vec2(XSIZE./DATA_XSIZE., YSIZE./DATA_YSIZE.);
	int op = int(oplane.x + oplane.y*16.);

	// calc w pos
	const vec2 arg = vec2(1./KERNEL_W., 1./KERNEL_W./KERNEL_H.);
	vec2 pos[4];
	pos[0] = arg * (float(op*4 *INPUTPLANE *9) +wpos +0.5);	// arg * (index+0.5)
	vec2 n = arg * float(INPUTPLANE *9);
	pos[1] = pos[0] + n;
	pos[2] = pos[1] + n;
	pos[3] = pos[2] + n;

	vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
	if (INPUTPLANE==1) {
		pos[0] = arg * (float(op *9) +wpos +0.5);	// arg * (index+0.5)

		vec4 p[9];
		p[0] = texture2D(X, a + vec2(-xSize, -ySize));
		p[1] = texture2D(X, a + vec2(   0.0, -ySize));
		p[2] = texture2D(X, a + vec2( xSize, -ySize));
		p[3] = texture2D(X, a + vec2(-xSize,    0.0));
		p[4] = texture2D(X, a + vec2(   0.0,    0.0));
		p[5] = texture2D(X, a + vec2( xSize,    0.0));
		p[6] = texture2D(X, a + vec2(-xSize,  ySize));
		p[7] = texture2D(X, a + vec2(   0.0,  ySize));
		p[8] = texture2D(X, a + vec2( xSize,  ySize));

		vec4 a[9];
		a[0] = texture2D(W, pos[0]); pos[0] += arg;	// 1-4
		a[1] = texture2D(W, pos[0]); pos[0] += arg;	// 5-8
		a[2] = texture2D(W, pos[0]); pos[0] += arg;	// 9-12
		a[3] = texture2D(W, pos[0]); pos[0] += arg;	// 13-16
		a[4] = texture2D(W, pos[0]); pos[0] += arg;	// 17-20
		a[5] = texture2D(W, pos[0]); pos[0] += arg;	// 21-24
		a[6] = texture2D(W, pos[0]); pos[0] += arg;	// 25-28
		a[7] = texture2D(W, pos[0]); pos[0] += arg;	// 29-32
		a[8] = texture2D(W, pos[0]);			// 33-36

		sum.x += dot(vec3(p[0].x, p[1].x, p[2].x), a[0].xyz);	// out 1
		sum.x += dot(vec3(p[3].x, p[4].x, p[5].x), vec3(a[0].w, a[1].x, a[1].y));
		sum.x += dot(vec3(p[6].x, p[7].x, p[8].x), vec3(a[1].z, a[1].w, a[2].x));

		sum.y += dot(vec3(p[0].x, p[1].x, p[2].x), a[2].yzw);	// out 2
		sum.y += dot(vec3(p[3].x, p[4].x, p[5].x), a[3].xyz);
		sum.y += dot(vec3(p[6].x, p[7].x, p[8].x), vec3(a[3].w, a[4].x, a[4].y));

		sum.z += dot(vec3(p[0].x, p[1].x, p[2].x), vec3(a[4].z, a[4].w, a[5].x));
		sum.z += dot(vec3(p[3].x, p[4].x, p[5].x), a[5].yzw);
		sum.z += dot(vec3(p[6].x, p[7].x, p[8].x), a[6].xyz);

		sum.w += dot(vec3(p[0].x, p[1].x, p[2].x), vec3(a[6].w, a[7].x, a[7].y));
		sum.w += dot(vec3(p[3].x, p[4].x, p[5].x), vec3(a[7].z, a[7].w, a[8].x));
		sum.w += dot(vec3(p[6].x, p[7].x, p[8].x), a[8].yzw);
	} else {
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
	}
	// Leaky ReLU
	sum += bias[op];
	sum = max(sum, 0.0) + min(sum, 0.0) * 0.1;
	gl_FragColor = sum;
//	gl_FragColor = texture2D(X, a);
//	if (op==2) gl_FragColor = texture2D(X, a+inputOffset[31]);
	//gl_FragColor = texture2D(W, arg * 71496.5);
	//gl_FragColor = texture2D(W, pos[0]);
	//gl_FragColor = texture2D(W, pos[1]);
	//gl_FragColor = bias[op];
	//if (op==16) gl_FragColor = vec4(1.,1.,1.,1.);	// 3,16
}

);

void *recalloc(void *p, int s, int ss)
{
	void *r = calloc(1, ss);
	if (!r) return 0;
	memcpy(r, p, s);
	free(p);
	return r;
}

//#define _DEBUG
#ifdef _DEBUG
#define debug_s(x)	{x;}
#else
#define debug_s(x)
#endif

void result(char *name, int w, int h)
{
	float *d = coReadDataf(w, h, 0);
#ifdef _DEBUG
	for (int i=0; i<8/*h*/; i++) {
		for (int j=0; j<8/*w*/; j++) printf("%2.3f ", d[(i*w+j)*4]);
		printf("\n");
	}
	printf("\n");
#endif

	uint8_t *o = calloc(w*h, 1);
	for (int y=0; y<h; y++) {
		for (int x=0; x<w; x++) {
			o[y*w+x] = d[(y*w+x)*4]*256;
			//o[y*w+x] = d[(y*w+x)*4+1]*256;
		}
	}
	stbi_write_png(name, w, h, 1, o, 0);
	free(o);
	free(d);
}

void waifu2x_glsl_run(CatsEye *cat, GLuint prog, GLuint *texture, float *yuv, uint8_t *s, int sx, int sy, uint8_t *p, int wx)
{
	float *u = yuv + 256*256*4;
	float *v = yuv + 256*256*5;
	int width = XSIZE;
	int height = YSIZE;
	if (sx<XSIZE) width = sx;	// small size <256
	if (sy<YSIZE) height = sy;
	for (int y=0; y<height; y++) {
		for (int x=0; x<width; x++) {
			uint8_t r = s[(y*sx+x)*3];
			uint8_t g = s[(y*sx+x)*3+1];
			uint8_t b = s[(y*sx+x)*3+2];

			yuv[(y*256+x)*4] = (0.298912*r +0.586611*g +0.114478*b)/256.0;	// CCIR Rec.601
			u[y*256+x] = -0.1687*r -0.3313*g +0.500 *b;
			v[y*256+x] =  0.500 *r -0.4187*g -0.0813*b;
//			yuv[(y*256+x)*4] = 0.299*r +0.587*g +0.114*b;	// CCIR Rec.601
//			u[y*256+x] = -0.147*r -0.289*g +0.436*b;
//			v[y*256+x] = 0.615*r -0.515*g -0.100*b;
		}
	}
//	debug_s(stbi_write_png("output_256.png", 256, 256, 3, p, 0));
//	debug_s(stbi_write_png("output_y.png", 256, 256, 1, yuv, 0));

	coTransferData(texture[0], 0, 0, XSIZE, YSIZE, GL_FLOAT, yuv);
	coBindInputTexture(prog, texture[2], GL_TEXTURE1, "W");

	debug_s(clock_start());
	int n = 0;
	int r = 1;
	for (int i=0; i<cat->layers; i++) {
		int a = (cat->u[i].out+3)/4;
		int w = a>16 ? 16 : a;
		int h = (a+15)/16;
		debug_s(printf("%d %d %dx%d %d %d %2.4f %2.4f\n", cat->u[i].in, cat->u[i].out, w, h, (cat->u[i].in+3)/4, cat->ws[i], cat->wdata[cat->ws[i]], cat->bdata[cat->bs[i]]));

		coUniform1i(prog, "INPUTPLANE", (cat->u[i].in+3)/4);
		coUniform4fv(prog, "bias", a, &cat->bdata[cat->bs[i]]); coAssert();
		coUniform2f(prog, "uvpos", (float)XSIZE*w/DATA_XSIZE, (float)YSIZE*h/DATA_YSIZE);
		coUniform1f(prog, "wpos", (float)cat->ws[i]/4);
		coBindInputTexture(prog, texture[n], GL_TEXTURE0, "X");
		coBindOutputTexture(XSIZE*w, YSIZE*h, texture[r]);
		coCompute();
		n ^= 1;	// swap
		r ^= 1;
#ifdef _DEBUG
		char buff[256];
		sprintf(buff, "output2x_%02d.png", i+1);
		result(buff, XSIZE*w, YSIZE*h);
#endif
	}
	debug_s(clock_end());

	float *d = coReadDataf(XSIZE, YSIZE, 0);
	for (int y=8; y<YSIZE-8; y++) {
		for (int x=8; x<XSIZE-8; x++) {
	//for (int y=0; y<YSIZE; y++) {
		//for (int x=0; x<XSIZE; x++) {
//			float yy = yuv[(y*256+x)*4];
			float yy = d[(y*256+x)*4]*256.0;
			int r = yy                     +1.402  *v[y*256+x];
			int g = yy -0.34414*u[y*256+x] -0.71414*v[y*256+x];
			int b = yy +1.772  *u[y*256+x];
			uint8_t *pix = &p[(y*wx+x)*3];
			if (!pix[0] || !pix[1] || !pix[2]) {
				pix[0] = r>255 ? 255 : r<0 ? 0 : r;
				pix[1] = g>255 ? 255 : g<0 ? 0 : g;
				pix[2] = b>255 ? 255 : b<0 ? 0 : b;
			}

//			p[(y*XSIZE+x)*3]   = 256*(yy                   +1.140*v[y*256+x]);
//			p[(y*XSIZE+x)*3+1] = 256*(yy -0.395*u[y*256+x] -0.580*v[y*256+x]);
//			p[(y*XSIZE+x)*3+2] = 256*(yy +2.032*u[y*256+x]);
		}
	}
	free(d);
}

int waifu2x_glsl(char *name, char *output, char *model, float scale)
{
	uint8_t *pixels;
	int w, h, bpp;
	pixels = stbi_load(name, &w, &h, &bpp, 3);
	assert(pixels);
	printf("%s %dx%d %d\n", name, w, h, bpp);
	bpp = 3;

	// resize
	int sx = w * scale;
	int sy = h * scale;
	uint8_t *pix = malloc(sx*sy*bpp);
	stbir_resize_uint8_srgb(pixels, w, h, 0, pix, sx, sy, 0, bpp, -1, 0);
	stbi_image_free(pixels);
	debug_s(stbi_write_jpg("output.jpg", sx, sy, bpp, pix, 0));

	// expand edge by +16
	sx += 16;
	sy += 16;
	pixels = calloc(sx*sy*bpp, 1);
//	pixels = calloc(sx*(sy+8)*bpp, 1);//FIXME
	for (int y=8; y<sy-8; y++) {
		memcpy(pixels +(8+(y*sx))*bpp, pix +((y-8)*(sx-16))*bpp, (sx-16)*bpp);
//		memcpy(pixels +y*sx*bpp, pix +((y-8)*(sx-16))*bpp, (sx-16)*bpp);
	}
	free(pix);
	pix = pixels;

	CatsEye cat;
	int r = CatsEye_loadJson(&cat, model);
	assert(!r);
	cat.wdata = recalloc(cat.wdata, sizeof(real)*cat.wsize, sizeof(real)*KERNEL_W*KERNEL_H*4); // 256*281
	cat.bdata = recalloc(cat.bdata, sizeof(real)*cat.bsize, sizeof(real)*(cat.bsize+3));

	coInit();
	GLuint prog = coCreateProgram(convolution);
	coBindVertices(prog);

	float ioffset[128/4*2];
	for (int i=0; i<128/4; i++) {
		ioffset[i*2] = (i % 16) / 16.0;
		ioffset[i*2+1] = floor(i / 16.0) / 8.0;
		//printf("%f %f\n", ioffset[i*2], ioffset[i*2+1]);
	}
	coUniform2fv(prog, "inputOffset", 128/4, ioffset);

	GLuint texture[3];
	texture[0] = coCreateDataTexture(DATA_XSIZE, DATA_YSIZE, 0, GL_FLOAT, 0);
//	coTransferData(texture[0], 0, 0, XSIZE, YSIZE, GL_FLOAT, f);
	texture[1] = coCreateDataTexture(DATA_XSIZE, DATA_YSIZE, 0, GL_FLOAT, 0);
	texture[2] = coCreateDataTexture(KERNEL_W, KERNEL_H, cat.wdata, GL_FLOAT, GPGPU_TEX_REPEAT);
//	coBindInputTexture(prog, texture[2], GL_TEXTURE1, "W");

	float *yuv = calloc(256*256*(4+2), sizeof(float));
//	uint8_t *o = calloc(XSIZE*YSIZE, 3);
//	waifu2x_glsl_run(&cat, prog, texture, yuv, pix, sx, sy, o, 256);
//	stbi_write_png("output2x.png", XSIZE, YSIZE, 3, o, 0);
	printf("%d %d -> %d %d *%f\n", w, h, sx, sy, scale);
	uint8_t *o = calloc(sx*sy, 3);
	for (int y=0; y<sy-1; y+=256-16) {
		for (int x=0; x<sx-1; x+=256-16) {
			int ox = x+256 > sx ? sx-(256+1) : x;
			int oy = y+256 > sy ? sy-(256+1) : y;
			printf("%d %d\n", ox, oy);
			waifu2x_glsl_run(&cat, prog, texture, yuv, pix+(ox+oy*sx)*3, sx, sy, o+(ox+oy*sx)*3, sx);
		}
	}
//	stbi_write_png(output, sx, sy, 3, o, 0);
//	free(o);
	free(yuv);
	free(pix);

	// shrink edge by -16
	sx -= 16;
	sy -= 16;
	pix = calloc(sx*sy*bpp, 1);
	for (int y=0; y<sy; y++) {
		memcpy(pix +(y*sx)*bpp, o +(8+(y+8)*(sx+16))*bpp, sx*bpp);
	}
	free(o);
	char *ext = strrchr(output, '.');
	if (!strcmp(ext, ".jpg")) stbi_write_jpg(output, sx, sy, 3, pix, 0);
	else stbi_write_png(output, sx, sy, 3, pix, 0);
	free(pix);

	free(cat.bdata);
	free(cat.wdata);

	coTerm();
	return 0;
}

void usage(FILE* fp, char** argv)
{
	fprintf(fp,
		"Usage: %s [options] file\n\n"
		"Options:\n"
		"-h                 Print this message\n"
		"-m <model name>    waifu2x model name [noise2_model.json...]\n"
		"-s <scale>         Magnification [1.0, 1.6, 2.0...]\n"
		"-o <output name>   output file name\n"
		"\n",
		argv[0]);
}

int main(int argc, char* argv[])
{
	char *name = 0;
	char *model = "noise1_model.json";
	char *output = "output2x.png";
	float scale = 2.0;

	struct parg_state ps;
	int c;
	parg_init(&ps);
	while ((c = parg_getopt(&ps, argc, argv, "hm:s:o:")) != -1) {
		switch (c) {
		case 1:
			name = (char*)ps.optarg;
			break;
		case 'o':
			output = (char*)ps.optarg;
			break;
		case 'm':
			model = (char*)ps.optarg;
			break;
		case 's':
			scale = atof(ps.optarg);
			break;
		case 'h':
		default:
			usage(stderr, argv);
			return 1;
		}
	}
	if (!name) {
		usage(stderr, argv);
		return 1;
	}
	waifu2x_glsl(name, output, model, scale);

	return 0;
}
