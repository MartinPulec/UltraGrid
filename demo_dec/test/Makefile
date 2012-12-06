# Demo decoder tester makefile

INCLUDES := -I /usr/local/cuda/include/
LIBS := -L /usr/local/cuda/lib64/

build: test.o
	cd .. && make
	g++ $(LIBS) -o test $< ../demo_dec.a -lpthread -lcudart

test.o: test.c ../demo_dec.h
	gcc $(INCLUDES) -O3 -Wno-long-long -std=c99 -pedantic -Wextra -c -o $@ $<

clean:
	rm -f *.o test
	cd .. && make clean
