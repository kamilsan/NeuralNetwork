CC=g++
CFLAGS= -std=c++17 -O3 -Wall -pedantic
CINCL= -I./include

program: main.o neuralnetwork.o layer.o reluLayer.o sigmoidLayer.o mnistDataLoader.o userInterface.o image.o matrix.h matrix.tpp
	${CC} ${CFLAGS} main.o neuralnetwork.o layer.o reluLayer.o sigmoidLayer.o mnistDataLoader.o userInterface.o image.o -o program

tests: tests.o neuralnetwork.o layer.o reluLayer.o sigmoidLayer.o mnistDataLoader.o userInterface.o image.o matrix.h matrix.tpp
	${CC} ${CFLAGS} tests.o neuralnetwork.o layer.o reluLayer.o sigmoidLayer.o mnistDataLoader.o userInterface.o image.o -o tests

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

sigmoidLayer.o: sigmoidLayer.cpp sigmoidLayer.h layer.cpp layer.h
	${CC} ${CFLAGS} -c sigmoidLayer.cpp -o sigmoidLayer.o

mnistDataLoader.o: mnistDataLoader.cpp mnistDataLoader.h mnistData.h
	${CC} ${CFLAGS} -c mnistDataLoader.cpp -o mnistDataLoader.o

userInterface.o: userInterface.cpp userInterface.h
	${CC} ${CFLAGS} -c userInterface.cpp -o userInterface.o

image.o: image.cpp image.h
	${CC} ${CFLAGS} -c image.cpp -o image.o

clean:
	rm -rf *.o program tests