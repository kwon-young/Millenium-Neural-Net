
#include "FNN_Model.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace Eigen;

FNN_Model::FNN_Model(
    std::vector<unsigned int> layers,
    unsigned int nbr_epoch,
    double learning_rate):
  _nbr_layer(layers.size()),
  _batch_size(1),
  _nbr_epoch(nbr_epoch),
  _learning_rate(learning_rate),
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
  _activations[0] = MatrixXd::Constant(_layers[0], _batch_size, 0);
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

void FNN_Model::SetInput(Eigen::MatrixXd &inputs)
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

double FNN_Model::sigmoidPrime(double input)
{
  return sigmoid(input)*(1-sigmoid(input));
}

void FNN_Model::FeedForward()
{
  for(unsigned int layer=1; layer<_nbr_layer; layer++)
  {
    _zs[layer] = _weights[layer] * _activations[layer-1] + _bias[layer];
    _activations[layer] = _zs[layer].unaryExpr(&FNN_Model::sigmoid);
  }
}

void FNN_Model::ComputeError(Eigen::MatrixXd &d_outputs)
{
  MatrixXd temp = _activations[_nbr_layer-1]-d_outputs;
  MatrixXd temp2 = _zs[_nbr_layer-1].unaryExpr(&FNN_Model::sigmoidPrime);
  _errors[_nbr_layer-1] = temp.cwiseProduct(temp2);
  for(int layer = _nbr_layer-2; layer >0; layer--)
  {
    temp = _weights[layer+1].transpose()*_errors[layer+1];
    temp2 = _zs[layer].unaryExpr(&FNN_Model::sigmoidPrime);
    _errors[layer] = temp.cwiseProduct(temp2);
  }
}

void FNN_Model::GradientDescent()
{
  double coeff = (_learning_rate/double(_batch_size));
  for(unsigned int layer=_nbr_layer-1; layer>0; layer--)
  {
    _weights[layer] -= coeff*_errors[layer]*_activations[layer-1].transpose();
    _bias[layer] -= coeff*_errors[layer].rowwise().sum();
  }
}

void FNN_Model::BackProgagation(
    Eigen::MatrixXd &inputs,
    Eigen::MatrixXd &d_outputs)
{
  SetInput(inputs);
  FeedForward();
  ComputeError(d_outputs);
  GradientDescent();
}

void FNN_Model::train(
    std::list<Eigen::VectorXd> &training_sample_i,
    std::list<Eigen::VectorXd> &training_sample_o,
    unsigned int batch_size)
{
  srand (time(NULL));
  unsigned int rand_pos = 0;
  MatrixXd inputs(_layers[0], batch_size);
  MatrixXd d_outputs(_layers[0], batch_size);
  for(unsigned int epoch=0; epoch<_nbr_epoch; epoch++)
  {
    while(training_sample_i.size() >0)
    {
      for(unsigned int i=0; i<batch_size; i++)
      {
        rand_pos = rand() % training_sample_i.size();
        //inputs.col(i) = training_sample_i.erase(training_sample_i.ite
      }
    }
  }
}

