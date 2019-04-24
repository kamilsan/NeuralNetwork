CC=g++
CFLAGS=-O3 -Wall -pedantic

program: main.o
	${CC} ${CFLAGS} main.o -o program

main.o: main.cpp matrix.h
	${CC} ${CFLAGS} -c main.cpp -o main.o

clean:
	rm -rf *.o program