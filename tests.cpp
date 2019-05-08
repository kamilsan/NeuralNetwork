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
    
    SECTION("adding two matricies") {
        Matrix<int> result = a+b;

        REQUIRE(result.get(0, 0) == 5);
        REQUIRE(result.get(0, 1) == 4);
        REQUIRE(result.get(1, 0) == 7);
        REQUIRE(result.get(1, 1) == 4);
    }

    SECTION("subtracting two matricies") {
        Matrix<int> result = a-b;

        REQUIRE(result.get(0, 0) == -3);
        REQUIRE(result.get(0, 1) == 0);
        REQUIRE(result.get(1, 0) == 1);
        REQUIRE(result.get(1, 1) == 2);
    }
}