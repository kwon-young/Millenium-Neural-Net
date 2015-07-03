
#include "Neural_Net.hpp"
#include <cmath>
#include <iostream>

using namespace Eigen;

double Neural_Net_Functions::sigmoid(double input)
{
   return 1.0 / (1.0 + exp(-input));
}


Neural_Net::Neural_Net(
      VectorXd &weight_sizes,
      Neural_Net_Functions *functions) :
   _layer_size(weight_sizes.size()),
   _weights(),
   _bias(),
   _activations()
{
   _bias.push_back(VectorXd::Random(weight_sizes[0]));
   _activations.push_back(VectorXd::Constant(weight_sizes[0], 0.0));
   for(int i = 1; i < _layer_size; i++)
   {
      _weights.push_back(MatrixXd::Random(weight_sizes[i], weight_sizes[i-1]));
      _bias.push_back(VectorXd::Random(weight_sizes[i]));
      _activations.push_back(VectorXd::Constant(weight_sizes[i], 0.0));
   }
}

Neural_Net::~Neural_Net()
{
}

void Neural_Net::vec_sigmoid(
      const MatrixXd &inputs,
      MatrixXd &outputs)
{
}

void Neural_Net::print_neural_net()
{
   for (unsigned int layer=0; layer<_layer_size; layer++)
   {
      std::cout << "layer number " << layer << std::endl;
      if (layer == 0)
      {
         std::cout << "input layer" << std::endl;
      } else
      {
         if (layer == _layer_size-1)
         {
            std::cout << "output layer" << std::endl;
         }
         std::cout << "weights" << std::endl;
         std::cout << _weights[layer-1] << std::endl;
      }
      std::cout << "bias" << std::endl;
      std::cout << _bias[layer] << std::endl;
         std::cout << "activation" << std::endl;
      std::cout << _activations[layer] << std::endl;
      std::cout << std::endl;
   }
}

