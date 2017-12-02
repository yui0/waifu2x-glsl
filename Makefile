# ©2017 YUICHIRO NAKADA

PROGRAM = waifu2x_glsl

CC	= clang
CPP	= clang++
#CFLAGS  = -Ofast -march=native -funroll-loops -mf16c -DDEBUG
CFLAGS  = -Ofast -march=native -funroll-loops -mf16c
CPPFLAGS= $(CFLAGS)
LDFLAGS	= -lm
CSRC	= $(wildcard *.c)
CPPSRC	= $(wildcard *.cpp)
DEPS	= $(wildcard *.h) Makefile
OBJS	= $(patsubst %.c,%.o,$(CSRC)) $(patsubst %.cpp,%.o,$(CPPSRC))

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
#	CFLAGS  += `pkg-config --cflags glesv2 egl gbm`
#	LDFLAGS	+= `pkg-config --libs glesv2 egl gbm` -lglfw
	CFLAGS  += `pkg-config --cflags gl`
	LDFLAGS	+= `pkg-config --libs gl` -lglfw
endif
ifeq ($(UNAME_S),Darwin)
	LDFLAGS	+= -framework OpenGL -lglfw
endif

%.o: %.cpp $(DEPS)
	$(CPP) -c -o $@ $< $(CPPFLAGS)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(PROGRAM): $(OBJS)
	$(CPP) -o $@ $^ -s $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) $(PROGRAM) $(OBJS) *.o *.s
