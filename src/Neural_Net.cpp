
#include "Neural_Net.hpp"
#include <cmath>
#include <iostream>

using namespace Eigen;

double Neural_Net_Functions::sigmoid(const double input) const
{
   return 1.0 / (1.0 + exp(-input));
}

double Neural_Net_Functions::cost(
    const Eigen::VectorXd &output,
    const Eigen::VectorXd &desired_output) const
{
  VectorXd temp = desired_output.array() - output.array();
  std::cout << "coucou" << std::endl;
  return 0.5*temp.squaredNorm();
}

Neural_Net::Neural_Net(
      VectorXd &weight_sizes,
      Neural_Net_Functions *functions) :
   _layer_size(weight_sizes.size()),
   _weights(),
   _bias(),
   _activations(),
   _zs(),
   _functions(functions)
{
   _activations.push_back(VectorXd::Constant(weight_sizes[0], 1.0));
   for(unsigned int i = 1; i < _layer_size; i++)
   {
      _weights.push_back(MatrixXd::Random(weight_sizes[i], weight_sizes[i-1]));
      _bias.push_back(VectorXd::Random(weight_sizes[i]));
      _activations.push_back(VectorXd::Constant(weight_sizes[i], 0.0));
      _zs.push_back(VectorXd::Constant(weight_sizes[i], 0.0));
   }
}

Neural_Net::~Neural_Net()
{
}

void Neural_Net::vec_sigmoid(unsigned int layer)
{
  //Eigen::VectorXd input = _weights[layer-1] * _activations[layer-1] + _bias[layer-1];
  _zs[layer-1] = _weights[layer-1] * _activations[layer-1] + _bias[layer-1];
  for (unsigned int i=0; i < _zs[layer-1].rows(); i++)
  {
    _activations[layer][i] = _functions->sigmoid(_zs[layer-1][i]);
  }
}

void Neural_Net::compute(Eigen::VectorXd input)
{
  _activations[0] = input;
  for (unsigned int layer = 1; layer < _layer_size; layer++)
  {
    vec_sigmoid(layer);
  }
}

double Neural_Net::getCost(Eigen::VectorXd desired_output) const
{
  return _functions->cost(_activations[_layer_size-1], desired_output);
}

void Neural_Net::print_layer(unsigned int layer) const
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
    std::cout << "bias" << std::endl;
    std::cout << _bias[layer-1] << std::endl;
    std::cout << "zs" << std::endl;
    std::cout << _zs[layer-1] << std::endl;
  }
  std::cout << "activation" << std::endl;
  std::cout << _activations[layer] << std::endl;
  std::cout << std::endl;
}

void Neural_Net::print_neural_net() const
{
   for (unsigned int layer=0; layer<_layer_size; layer++)
   {
     print_layer(layer);
   }
}

