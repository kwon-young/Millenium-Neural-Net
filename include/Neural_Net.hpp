
#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <Eigen/Dense>
#include <vector>

//class Sigmoid_Neuron
//{
   //Sigmoid_Neuron(int input_size, 
   //private:
      //Eigen::VectorXd weights;
      //double activation;
      //double bias;
//};

class Neural_Net_Functions
{
   public:
      Neural_Net_Functions() {}
      ~Neural_Net_Functions() {}

      virtual double sigmoid(const double input) const;
      //virtual double cost(double output, double desired_output);
};


class Neural_Net
{
   public:
      Neural_Net(
            Eigen::VectorXd &weight_sizes,
            Neural_Net_Functions *functions);
      ~Neural_Net();

      void vec_sigmoid(unsigned int layer);
      void compute(Eigen::VectorXd input);

      void print_neural_net();
      void print_layer(unsigned int layer);

   private:
      unsigned int _layer_size;
      std::vector<Eigen::MatrixXd> _weights;
      std::vector<Eigen::VectorXd> _bias;
      std::vector<Eigen::VectorXd> _activations;
      Neural_Net_Functions *_functions;
};

#endif

