#pragma once

#include "matrix.h"
#include "neuralnetwork.h"

#include <fstream>

class CostFunctionStrategy
{
public:
    //Abstract cost function
    virtual NNDataType calculateCost(const NNMatrixType& output, const NNMatrixType& target) const = 0;
    
    //Derivative of abstract cost function 
    virtual NNMatrixType calculateCostDerivative(const NNMatrixType& output, const NNMatrixType& target) const = 0;

    //Abstract serialization method
    virtual void serialize(std::ofstream& ofile) const = 0;
};