#include "crossEntropyCost.hpp"

#include "matrix.hpp"

NNDataType CrossEntropyCost::calculateCost(const NNMatrixType& output, const NNMatrixType& target) const
{
    NNDataType error = 0;
    for(unsigned int i = 0; i < target.getRows(); ++i)
    {
        NNDataType ai = output.get(i, 0);
        NNDataType yi = target.get(i, 0);
        error += -yi*std::log(ai) - (1.0 - yi)*std::log(1.0 - ai);
    }
    return error;
}

NNMatrixType CrossEntropyCost::calculateCostDerivative(const NNMatrixType& output, const NNMatrixType& target) const
{
    // dc/da = (a-y)/(a(1-a))
    unsigned int rows = output.getRows();
    NNMatrixType result = NNMatrixType(rows, 1);

    for(unsigned int i = 0; i < rows; ++i)
    {
        NNDataType ai = output.get(i, 0);
        NNDataType yi = target.get(i, 0);
        result[i] = (ai - yi)/(ai*(1.0 - ai));
    }

    return result;
}

void CrossEntropyCost::serialize(std::ofstream& ofile) const
{
    const char* id = "CEX";
    const unsigned int ID_SIZE = 3;
    ofile.write((char*)&ID_SIZE, sizeof(ID_SIZE));
    ofile.write(id, ID_SIZE*sizeof(char));
}
