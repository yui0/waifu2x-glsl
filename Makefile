# ©2017 YUICHIRO NAKADA

PROGRAM = waifu2x_glsl

CC	= clang
CPP	= clang++
CFLAGS	= -Wall -Os `pkg-config --cflags glesv2 egl gbm`
CPPFLAGS= $(CFLAGS)
LDFLAGS	= `pkg-config --libs glesv2 egl gbm` -lglfw -lm
CSRC	= $(wildcard *.c)
CPPSRC	= $(wildcard *.cpp)
DEPS	= $(wildcard *.h) Makefile
OBJS	= $(patsubst %.c,%.o,$(CSRC)) $(patsubst %.cpp,%.o,$(CPPSRC))

%.o: %.cpp $(DEPS)
	$(CPP) -c -o $@ $< $(CPPFLAGS)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(PROGRAM): $(OBJS)
	$(CPP) -o $@ $^ -s $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) $(PROGRAM) $(OBJS) *.o *.s
