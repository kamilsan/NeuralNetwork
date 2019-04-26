CC=g++
CFLAGS=-O3 -Wall -pedantic

program: main.o neuralnetwork.o
	${CC} ${CFLAGS} main.o -o program

main.o: main.cpp matrix.h
	${CC} ${CFLAGS} -c main.cpp -o main.o

neuralnetwork.o: neuralnetwork.cpp neuralnetwork.h
	${CC} ${CFLAGS} -c neuralnetwork.cpp -o neuralnetwork.o

clean:
	rm -rf *.o program