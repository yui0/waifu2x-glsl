/* public domain Simple, Minimalistic, clock library
 *	©2017 Yuichiro Nakada
 * */

#include <time.h>
#include <sys/time.h>

#define ANSI_COLOR_RED		"\x1b[31m"
#define ANSI_COLOR_GREEN	"\x1b[32m"
#define ANSI_COLOR_YELLOW	"\x1b[33m"
#define ANSI_COLOR_BLUE		"\x1b[34m"
#define ANSI_COLOR_MAGENTA	"\x1b[35m"
#define ANSI_COLOR_CYAN		"\x1b[36m"
#define ANSI_COLOR_RESET	"\x1b[0m"

struct timeval __t0;
clock_t __startClock;

void clock_start()
{
	gettimeofday(&__t0, NULL);
	__startClock = clock();
}

void clock_end()
{
	struct timeval t1;
	clock_t endClock;

	gettimeofday(&t1, NULL);
	endClock = clock();
	time_t diffsec = difftime(t1.tv_sec, __t0.tv_sec);
	suseconds_t diffsub = t1.tv_usec - __t0.tv_usec;
	double realsec = diffsec+diffsub*1e-6;
	double cpusec = (endClock - __startClock)/(double)CLOCKS_PER_SEC;
	double percent = 100.0*cpusec/realsec;
	printf(ANSI_COLOR_MAGENTA "Time spent on GPU: %f\n" ANSI_COLOR_RESET, realsec);
	printf(ANSI_COLOR_GREEN "CPU utilization: %f\n" ANSI_COLOR_RESET, cpusec);
}
