#include "reluLayer.hpp"

NNDataType ReLULayer::activationFunction(NNDataType value) const
{
    if(value < 0.0f) return 0.0f;
    else return value;
}

NNDataType ReLULayer::activationDerivative(NNDataType value) const
{
    if(value < 0.0f) return 0.0f;
    else return 1.0f;
}

void ReLULayer::serialize(std::ofstream& ofile) const
{
    const char* id = "REL";
    const unsigned int ID_LEN = 3;
    ofile.write((char*)&ID_LEN, sizeof(ID_LEN));
    ofile.write(id, ID_LEN*sizeof(char));
    
    serializeMatricies(ofile);
}