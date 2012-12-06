

cuda_postproc_v210.o: cuda_postproc_v210.cu cuda_postproc_v210.h
	nvcc -I/usr/local/cuda/include -O3 --ptxas-options="-v" \
	     -gencode=arch=compute_30,code=\"sm_30,compute_30\" \
	     -gencode=arch=compute_20,code=\"sm_20,compute_20\" \
	     -c -o $@ $<

tools: postproc_test v210_to_tga

postproc_test: postproc_test.o cuda_postproc_v210.o 
	nvcc -o $@ $^

postproc_test.o: postproc_test.c cuda_postproc_v210.h
	gcc -O3 -std=c89 -pedantic -Wextra -I/usr/local/cuda/include \
            -Wno-long-long -c -o $@ $<

v210_to_tga: v210_to_tga.c
	gcc -std=c99 -pedantic -Wextra -o $@ $<

clean:
	rm -f *.o postproc_test v210_to_tga
