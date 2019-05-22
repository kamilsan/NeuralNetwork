CC=g++
CFLAGS= -std=c++17 -O3 -Wall -pedantic

program: main.o neuralnetwork.o layer.o reluLayer.o mnistDataLoader.o userInterface.o image.o
	${CC} ${CFLAGS} main.o neuralnetwork.o layer.o reluLayer.o mnistDataLoader.o userInterface.o image.o -o program

tests: tests.o neuralnetwork.o layer.o reluLayer.o mnistDataLoader.o userInterface.o image.o
	${CC} ${CFLAGS} tests.o neuralnetwork.o layer.o reluLayer.o mnistDataLoader.o userInterface.o image.o -o tests

tests.o: tests.cpp
	${CC} ${CFLAGS} ${CINCL} -c tests.cpp -o tests.o

main.o: main.cpp
	${CC} ${CFLAGS} -c main.cpp -o main.o

neuralnetwork.o: neuralnetwork.cpp neuralnetwork.h
	${CC} ${CFLAGS} -c neuralnetwork.cpp -o neuralnetwork.o

layer.o: layer.cpp layer.h
	${CC} ${CFLAGS} -c layer.cpp -o layer.o

reluLayer.o: reluLayer.cpp reluLayer.h layer.cpp layer.h
	${CC} ${CFLAGS} -c reluLayer.cpp -o reluLayer.o

mnistDataLoader.o: mnistDataLoader.cpp mnistDataLoader.h mnistData.h
	${CC} ${CFLAGS} -c mnistDataLoader.cpp -o mnistDataLoader.o

clean:
	rm -rf *.o program