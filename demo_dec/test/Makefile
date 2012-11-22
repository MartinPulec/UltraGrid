# Demo decoder tester makefile

INCLUDES := -I /usr/local/cuda/include/
LIBS := -L /usr/local/cuda/lib64/

test: test.o ../demo_dec.a
	g++ $(LIBS) -o test $^ -lpthread -lcudart

../demo_dec.a: 
	cd .. && make

test.o: test.c ../demo_dec.h
	gcc $(INCLUDES) -O3 -Wno-long-long -std=c89 -pedantic -Wextra -c -o $@ $<

clean:
	rm -f *.o test
