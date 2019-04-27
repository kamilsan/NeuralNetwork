CC=g++
CFLAGS= -std=c++17 -O3 -Wall -pedantic

program: main.o neuralnetwork.o mnistDataLoader.o
	${CC} ${CFLAGS} main.o -o program

main.o: main.cpp
	${CC} ${CFLAGS} -c main.cpp -o main.o

neuralnetwork.o: neuralnetwork.cpp neuralnetwork.h
	${CC} ${CFLAGS} -c neuralnetwork.cpp -o neuralnetwork.o

mnistDataLoader.o: mnistDataLoader.cpp mnistDataLoader.h mnistData.h
	${CC} ${CFLAGS} -c mnistDataLoader.cpp -o mnistDataLoader.o

clean:
	rm -rf *.o program