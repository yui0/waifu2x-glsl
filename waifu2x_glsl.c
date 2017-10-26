// clang -Os waifu2x_glsl.c -o waifu2x_glsl `pkg-config --libs --cflags glesv2 egl gbm` -lglfw -lm
#include <stdlib.h>
#include "gpgpu_glsl.h"
#define PARSON_IMPLEMENTATION
#include "parson.h"
//#define YUVRGB_IMPLEMENTATION
//#include "yuv_rgb.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define numerus		float
typedef struct {
	// number of each layer
	int layers, *u;

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
	this->u = malloc(sizeof(int)*7*this->layers);
	this->b = malloc(sizeof(numerus*)*(this->layers));
	this->bs = malloc(sizeof(int)*(this->layers));
	this->w = malloc(sizeof(numerus*)*(this->layers));
	this->ws = malloc(sizeof(int)*(this->layers));

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

uniform int INPUTPLANE;// INPUTPLANE/4
uniform vec2 inputOffset[128/4];
uniform float windex;	// +0.5

uniform sampler2D X;
uniform sampler2D W;
uniform vec4 bias;	// 4

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
	const vec2 arg = vec2(1./KERNEL_W., 1./KERNEL_W./KERNEL_H.);// arg = vec2(1./size.x, 1./size.x/size.y);
	vec2 pos[4];
	pos[0] = arg * windex;	// arg * (index+0.5)
	vec2 n = arg * float(INPUTPLANE *9);
	pos[1] = pos[0] + n;
	pos[2] = pos[1] + n;
	pos[3] = pos[2] + n;

	vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
	for (int i=0; i<INPUTPLANE; i++) {
		vec2 tuv = uv*vec2(XSIZE./DATA_XSIZE., YSIZE./DATA_YSIZE.) + inputOffset[i];

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
	sum += bias;
	sum = max(sum, 0.0) + min(sum, 0.0) * 0.1;
	gl_FragColor = sum;
	//gl_FragColor = texture2D(X, uv*vec2(XSIZE./DATA_XSIZE., YSIZE./DATA_YSIZE.));
}

);

/*void resizer(int argc, char **argv)
{
	unsigned char* input_pixels;
	unsigned char* output_pixels;
	int w, h;
	int n;
	int out_w, out_h;
	input_pixels = stbi_load(argv[1], &w, &h, &n, 0);
	out_w = w*3;
	out_h = h*3;
	output_pixels = (unsigned char*) malloc(out_w*out_h*n);
	//stbir_resize_uint8_srgb(input_pixels, w, h, 0, output_pixels, out_w, out_h, 0, n, -1,0);
	//stbir_resize_uint8(input_pixels, w, h, 0, output_pixels, out_w, out_h, 0, n);
	stbir_resize_float(input_pixels, w, h, 0, output_pixels, out_w, out_h, 0, n);
	stbi_write_png("output.png", out_w, out_h, n, output_pixels, 0);
	exit(0);
}*/
void convert_image_float(const unsigned char* input, float* output, int length)
{
	for (int i=0; i<length; i++) output[i] = ((float)input[i])/255;
}

void *recalloc(void *p, int s, int ss)
{
	void *r = calloc(ss, 1);
	if (!r) return 0;
	memcpy(r, p, s);
	free(p);
	return r;
}

int32_t main(int32_t argc, char* argv[])
{
	unsigned char *pixels;
	int w, h, bpp;
	pixels = stbi_load(argv[1], &w, &h, &bpp, 3);
	printf("%s %dx%d %d\n", argv[1], w, h, bpp);

	unsigned char *pix = malloc(w*2*h*2*bpp);
	stbir_resize_uint8_srgb(pixels, w, h, 0, pix, w*2, h*2, 0, bpp, -1, 0);
	//stbir_resize_uint8(pixels, w, h, 0, pix, w*2, h*2, 0, bpp);
	//float *pix = malloc(sizeof(float)*256*256);
	//stbir_resize_float(pixels, w, h, 0, pix, 256, 256, 0, 1);
	stbi_image_free(pixels);
//	free(pix);
	//stbi_write_png("output.png", w*2, h*2, bpp, pix, 0);
	stbi_write_jpg("output.jpg", w*2, h*2, bpp, pix, 0);
//	exit(0);

	unsigned char *p = malloc(256*256*3);
	unsigned char *yuv = calloc(256*256*3/2, 1);
	unsigned char *y = yuv;
	unsigned char *u = yuv +256*256;
	unsigned char *v = yuv +256*256 +((256+1)/2)*((256+1)/2);
	float *f = calloc(256*256*4, sizeof(float));
	for (int y=0; y<256; y++) {
		for (int x=0; x<256; x++) {
//			p[(y*256+x)*3] = pix[(y*w*2+x)*3];
//			p[(y*256+x)*3+1] = pix[(y*w*2+x)*3+1];
//			p[(y*256+x)*3+2] = pix[(y*w*2+x)*3+2];
			unsigned char r = pix[(y*w*2+x)*3];
			unsigned char g = pix[(y*w*2+x)*3+1];
			unsigned char b = pix[(y*w*2+x)*3+2];
			p[(y*256+x)*3] = 0.299*r +0.587*g +0.114*b;
			p[(y*256+x)*3+1] = -0.169*r -0.331*g +0.500*b;
			p[(y*256+x)*3+2] = 0.500*r -0.419*g -0.081*b;
			yuv[y*256+x] = 0.299*r +0.587*g +0.114*b;
			f[(y*256+x)*4] = (0.299*r +0.587*g +0.114*b)/255.0;
		}
	}
	stbi_write_png("output_256.png", 256, 256, 3, p, 0);
	//rgb24_yuv420_sseu(256, 256, p, 0, y, u, v, 0, 0, YCBCR_JPEG);
	//rgb24_yuv420_std(256, 256, p, 0, y, u, v, 0, 0, YCBCR_JPEG);
//	stbi_write_png("output.png", 256, 256, 1, y, 0);
	stbi_write_png("output_y.png", 256, 256, 1, y, 0);
	free(p);

	CatsEye cat;
	CatsEye_loadJson(&cat, "noise1_model.json");
	cat.wdata = recalloc(cat.wdata, cat.wsize, KERNEL_W*KERNEL_H*sizeof(numerus)*2);
	cat.bdata = recalloc(cat.bdata, cat.bsize, cat.bsize+3);

	coInit();

	GLuint prog = coCreateProgram(convolution);
	coBindVertices(prog);

	GLuint texture0 = coCreateDataTexture(DATA_YSIZE, DATA_XSIZE, 0, GL_FLOAT);
	coTransferData(texture0, 0, 0, XSIZE, YSIZE, GL_FLOAT, f);
	GLuint texture1 = coCreateDataTexture(KERNEL_H, KERNEL_W, cat.wdata, GL_FLOAT);
	GLuint texture3 = coCreateDataTexture(DATA_YSIZE, DATA_XSIZE, 0, GL_FLOAT);
	coBindInputTexture(prog, texture0, GL_TEXTURE0, "X");
	coBindInputTexture(prog, texture1, GL_TEXTURE1, "W");

	float ioffset[128/4*2];
	for (int i=0; i<128/4*2; i++) {
                ioffset[i*2] = (i % 16) / 16.0;
                ioffset[i*2+1] = floor(i / 16.0) / 8.0;
		//printf("%f %f\n", ioffset[i*2], ioffset[i*2+1]);
	}
	coUniform2fv(prog, "inputOffset", 128/4*2, ioffset);

	coUniform4fv(prog, "bias", 1, cat.bdata); coAssert();
	coUniform1i(prog, "INPUTPLANE", 1);
	coUniform1f(prog, "windex", 0.5);
	coBindOutputTexture(YSIZE, XSIZE, texture3);
	coCompute();

	float *d = coReadDataf(YSIZE, XSIZE, 0);
	for (int i=0; i<YSIZE; i++) {
//		for (int j=0; j<XSIZE*4; j++) printf("%2.2f ", d[i*XSIZE*4+j]);
		for (int j=0; j<XSIZE; j++) printf("%2.2f ", d[(i*XSIZE+j)*4]);
		printf("\n");
	}
	printf("\n");

	/*unsigned char *dd = (unsigned char*)d;
	for (int i=0; i<YSIZE*XSIZE*4; i++) printf("%x ", dd[i]);
	printf("\n");*/

	unsigned char *o = calloc(256*256*3, 1);
	for (int y=0; y<256; y++) {
		for (int x=0; x<256; x++) {
			o[(y*256+x)*3] = d[(y*XSIZE+x)*4]*255;
			//o[(y*256+x)*3] = d[(y*256+x)*4+1]*255;
			//o[(y*256+x)*3] = d[(y*256+x)*4+2]*255;
			//o[(y*256+x)*3] = d[(y*256+x)*4+3]*255;
		}
	}
	stbi_write_png("output_2x.png", 256, 256, 3, o, 0);
	free(o);
	free(yuv);
	free(d);

	coTerm();
	return 0;
}
