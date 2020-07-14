#include "sigmoidLayer.hpp"

NNDataType SigmoidLayer::activationFunction(NNDataType value) const
{
    return 1.0/(1.0 + exp(-value));
}

NNDataType SigmoidLayer::activationDerivative(NNDataType value) const
{
    NNDataType sigmoid = 1.0/(1.0 + exp(-value));
    return sigmoid*(1.0-sigmoid);
}

void SigmoidLayer::serialize(std::ofstream& ofile) const
{
    const char* id = "SIG";
    const unsigned int ID_LEN = 3;
    ofile.write((char*)&ID_LEN, sizeof(ID_LEN));
    ofile.write(id, ID_LEN*sizeof(char));

    serializeMatricies(ofile);
}
