
#include "Neural_Net.hpp"
#include <cmath>
#include <iostream>

using namespace Eigen;

double Neural_Net_Functions::logistic(const double input) const
{
   return 1.0 / (1.0 + exp(-input));
}

double Neural_Net_Functions::logisticPrime(const double input) const
{
  return logistic(input)*(1-logistic(input));
}

void Neural_Net_Functions::vec_logistic(
    const Eigen::VectorXd &input,
    Eigen::VectorXd &output) const
{
  for (unsigned int i=0; i < output.rows(); i++)
  {
    output[i] = logistic(input[i]);
  }
}

void Neural_Net_Functions::vec_logisticPrime(
    const Eigen::VectorXd &input,
    Eigen::VectorXd &output) const
{
  for (unsigned int i=0; i < output.rows(); i++)
  {
    output[i] = logisticPrime(input[i]);
  }
}

double Neural_Net_Functions::cost(
    const Eigen::VectorXd &output,
    const Eigen::VectorXd &desired_output) const
{
  VectorXd temp = desired_output.array() - output.array();
  return 0.5*temp.squaredNorm();
}

Eigen::VectorXd Neural_Net_Functions::costPrime(
    const Eigen::VectorXd &output,
    const Eigen::VectorXd &desired_output) const
{
  return desired_output.array() - output.array();
}

Neural_Net::Neural_Net(
      VectorXd &weight_sizes,
      Neural_Net_Functions *functions) :
   _layer_size(weight_sizes.size()),
   _weights(),
   _bias(),
   _activations(),
   _zs(),
   _errors(),
   _gradientSum(0.0),
   _functions(functions)
{
   _activations.push_back(VectorXd::Constant(weight_sizes[0], 1.0));
   for(unsigned int i = 1; i < _layer_size; i++)
   {
      _weights.push_back(MatrixXd::Random(weight_sizes[i], weight_sizes[i-1]));
      _bias.push_back(VectorXd::Random(weight_sizes[i]));
      _activations.push_back(VectorXd::Constant(weight_sizes[i], 0.0));
      _zs.push_back(VectorXd::Constant(weight_sizes[i], 0.0));
      _errors.push_back(VectorXd::Constant(weight_sizes[i], 0.0));
   }
}

Neural_Net::~Neural_Net()
{
}

void Neural_Net::feedForward()
{
  for (unsigned int layer = 1; layer < _layer_size; layer++)
  {
    _zs[layer-1] = _weights[layer-1] * _activations[layer-1] + _bias[layer-1];
    _functions->vec_logistic(_zs[layer-1], _activations[layer]);
  }
}

double Neural_Net::getCost(Eigen::VectorXd desired_output) const
{
  return _functions->cost(_activations[_layer_size-1], desired_output);
}

void Neural_Net::computeError(Eigen::VectorXd desired_output)
{
  VectorXd temp = _functions->costPrime(
      _activations[_layer_size-1],
      desired_output);
  _functions->vec_logisticPrime(_zs[_layer_size-2], _zs[_layer_size-2]);
  _errors[_layer_size-2] = temp.cwiseProduct(_zs[_layer_size-2]);
  for (unsigned int layer = _layer_size-2; layer > 0; layer--)
  {
    VectorXd temp = _weights[layer].transpose() * _errors[layer];
    _functions->vec_logisticPrime(_zs[layer-1], _zs[layer-1]);
    _errors[layer-1] = temp.cwiseProduct(_zs[layer-1]);
  }
}

void Neural_Net::gradientSum()
{
   //_gradientSum += _activations[
}

void Neural_Net::backPropagation(
    Eigen::MatrixXd input,
    Eigen::MatrixXd desired_output)
{
  for (unsigned int i=0; i < input.cols(); i++)
  {
    setInput(input.col(i));
    feedForward();
    computeError(desired_output.col(i));
  }
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
    std::cout << "errors" << std::endl;
    std::cout << _errors[layer-1] << std::endl;
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

