#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "matrix.h"
#include "mnistDataLoader.h"
#include "neuralnetwork.h"
#include "userInterface.h"

TEST_CASE("matrix operations can be performed", "[matrix]") 
{
    int aData[] = {1, 2, 4, 3};
    int bData[] = {4, 2, 3, 1};

    Matrix<int> a = Matrix<int>(aData, 2, 2);
    Matrix<int> b = Matrix<int>(bData, 2, 2);

    REQUIRE(a.getRows() == 2);
    REQUIRE(a.getColumns() == 2);
    REQUIRE(b.getRows() == 2);
    REQUIRE(b.getColumns() == 2);
    
    SECTION("adding two matricies")
    {
        auto result = a+b;

        REQUIRE(result.get(0, 0) == 5);
        REQUIRE(result.get(0, 1) == 4);
        REQUIRE(result.get(1, 0) == 7);
        REQUIRE(result.get(1, 1) == 4);
    }

    SECTION("subtracting two matricies")
    {
        auto result = a-b;

        REQUIRE(result.get(0, 0) == -3);
        REQUIRE(result.get(0, 1) == 0);
        REQUIRE(result.get(1, 0) == 1);
        REQUIRE(result.get(1, 1) == 2);
    }

    SECTION("sum of matrix entries")
    {
        REQUIRE(a.sum() == 10);
    }

    SECTION("transpose")
    {
        auto result = Matrix<int>::transpose(a);

        REQUIRE(result.get(0, 0) == 1);
        REQUIRE(result.get(0, 1) == 4);
        REQUIRE(result.get(1, 0) == 2);
        REQUIRE(result.get(1, 1) == 3);
    }

    SECTION("matrix multiplication")
    {
        auto result = a*b;

        REQUIRE(result.get(0, 0) == 10);
        REQUIRE(result.get(0, 1) == 4);
        REQUIRE(result.get(1, 0) == 25);
        REQUIRE(result.get(1, 1) == 11);
    }

    SECTION("hadamard product")
    {
        auto result = a.hadamard(b);

        REQUIRE(result.get(0, 0) == 4);
        REQUIRE(result.get(0, 1) == 4);
        REQUIRE(result.get(1, 0) == 12);
        REQUIRE(result.get(1, 1) == 3);
    }
}

TEST_CASE("saving and loading neural network", "[nn]")
{
    NeuralNetwork nn = NeuralNetwork(10, 20, 5, 0.1);

    float matrixData[] = {1, 2, 3, 1, 2, 3, 6, 3, 1, 2};
    NNMatrixType input(matrixData, 10, 1);
    NNMatrixType result;

    nn.feedforward(input, result);

    nn.save("nn_test.model");

    NeuralNetwork* nn2 = NeuralNetwork::load("nn_test.model");
    NNMatrixType result2;
    nn2->feedforward(input, result2);

    delete nn2;

    const float EPS = 0.00001;

    REQUIRE(fabs(result.get(0, 0) - result2.get(0, 0)) < EPS);
    REQUIRE(fabs(result.get(1, 0) - result2.get(1, 0)) < EPS);
    REQUIRE(fabs(result.get(2, 0) - result2.get(2, 0)) < EPS);
    REQUIRE(fabs(result.get(3, 0) - result2.get(3, 0)) < EPS);
    REQUIRE(fabs(result.get(4, 0) - result2.get(4, 0)) < EPS);
}