CC=g++
CFLAGS= -std=c++17 -O3 -Wall -pedantic

program: main.o neuralnetwork.o mnistDataLoader.o userInterface.o image.o
	${CC} ${CFLAGS} main.o neuralnetwork.o mnistDataLoader.o userInterface.o image.o -o program

main.o: main.cpp
	${CC} ${CFLAGS} -c main.cpp -o main.o

neuralnetwork.o: neuralnetwork.cpp neuralnetwork.h
	${CC} ${CFLAGS} -c neuralnetwork.cpp -o neuralnetwork.o

mnistDataLoader.o: mnistDataLoader.cpp mnistDataLoader.h mnistData.h
	${CC} ${CFLAGS} -c mnistDataLoader.cpp -o mnistDataLoader.o

userInterface.o: userInterface.cpp userInterface.h
	${CC} ${CFLAGS} -c userInterface.cpp -o userInterface.o

image.o: image.cpp image.h
	${CC} ${CFLAGS} -c image.cpp -o image.o

clean:
	rm -rf *.o program