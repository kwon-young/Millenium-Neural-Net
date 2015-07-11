
#include "FNN_Model.hpp"
#include <iostream>
#include <cmath>

using namespace Eigen;

FNN_Model::FNN_Model(
    std::vector<unsigned int> layers,
    unsigned int nbr_epoch,
    unsigned int batch_size):
  _nbr_layer(layers.size()),
  _batch_size(batch_size),
  _nbr_epoch(nbr_epoch),
  _layers(layers),
  _weights(_nbr_layer),
  _bias(_nbr_layer),
  _zs(_nbr_layer),
  _activations(_nbr_layer),
  _errors(_nbr_layer)
{
  Init();
}

FNN_Model::~FNN_Model()
{
}

void FNN_Model::print_FNN()
{
  for(unsigned int layer=0; layer<_nbr_layer; layer++)
  {
    std::cout << "Layer number " << layer << std::endl;
    if (layer==0)
    {
      std::cout << "Activation" << std::endl;
      std::cout << _activations[layer] << std::endl;
    } else {
      std::cout << "Weights" << std::endl;
      std::cout << _weights[layer] << std::endl;
      std::cout << "Bias" << std::endl;
      std::cout << _bias[layer] << std::endl;
      std::cout << "Zs" << std::endl;
      std::cout << _zs[layer] << std::endl;
      std::cout << "Activation" << std::endl;
      std::cout << _activations[layer] << std::endl;
      std::cout << "error" << std::endl;
      std::cout << _errors[layer] << std::endl;
    }
  }
}

void FNN_Model::Init()
{
  _activations[0] = MatrixXd::Random(_layers[0], 1);
  for(unsigned int layer=1; layer<_nbr_layer; layer++)
  {
    _weights[layer] = MatrixXd::Random(
        _layers[layer],
        _layers[layer-1]);
    _bias[layer] = MatrixXd::Random(_layers[layer], _batch_size);
    _zs[layer] = MatrixXd::Constant(_layers[layer], _batch_size, 0);
    _activations[layer] = MatrixXd::Constant(_layers[layer], _batch_size, 0);
    _errors[layer] = MatrixXd::Constant(_layers[layer], _batch_size, 0);
  }
}

void FNN_Model::ResizeBatch()
{
  for(unsigned int layer=1; layer<_nbr_layer; layer++)
  {
    MatrixXd temp = _bias[layer].col(0);
    _bias[layer].resize(_layers[layer], _batch_size);
    for(unsigned int i=0; i<_bias[layer].cols(); i++)
    {
      _bias[layer].col(i) << temp;
    }
    _zs[layer].resize(_layers[layer], _batch_size);
    _activations[layer].resize(_layers[layer], _batch_size);
    _errors[layer].resize(_layers[layer], _batch_size);
  }
}

void FNN_Model::SetInput(Eigen::MatrixXd inputs)
{
  _activations[0] = inputs;
  if (inputs.cols() != _batch_size)
  {
    _batch_size = inputs.cols();
    ResizeBatch();
  }
}

double FNN_Model::sigmoid(double input)
{
  return 1.0/(1.0+exp(-input));
}

void FNN_Model::FeedForward()
{
  for(unsigned int layer=1; layer<_nbr_layer; layer++)
  {
    _zs[layer] = _weights[layer] * _activations[layer-1] + _bias[layer];
    _activations[layer] = _zs[layer].unaryExpr(&FNN_Model::sigmoid);
  }
}

