CU_APPS=test

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -D_FORCE_INLINES -o $@ $<
%: %.c
	gcc -O2 -std=c99 -D_FORCE_INLINES -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
