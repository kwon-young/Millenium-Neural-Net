
#ifndef FNN_MODEL_HPP
#define FNN_MODEL_HPP

#include <vector>
#include <list>
#include <Eigen/Dense>
#include "SFML/System.hpp"
#include "gnuplot_i.hpp"

struct Data
{
  Eigen::MatrixXd &training_sample_i;
  Eigen::MatrixXd &training_sample_o;
  Eigen::MatrixXd &eval_input;
  Eigen::MatrixXd &eval_output;
  Data(Eigen::MatrixXd &t_i, Eigen::MatrixXd &t_o,
      Eigen::MatrixXd &e_i, Eigen::MatrixXd &e_o) :
    training_sample_i(t_i),
    training_sample_o(t_o),
    eval_input(e_i),
    eval_output(e_o)
  {}
};
class FNN_Model
{
  public:
    FNN_Model(
        std::vector<unsigned int> layers);
    ~FNN_Model();

    void print_FNN();
    static double normal_distri(double input);
    void Manual_Set_FNN(
        std::vector<Eigen::MatrixXd> &weights,
        std::vector<Eigen::MatrixXd> &bias);
    void Init();
    void ResizeBatch();
    void SetInput(Eigen::MatrixXd &inputs);
    static double sigmoid(double input);
    static double sigmoidPrime(double input);
    void FeedForward();
    void ComputeError(Eigen::MatrixXd &d_outputs);
    void GradientDescent();
    void BackProgagation(
        Eigen::MatrixXd &inputs,
        Eigen::MatrixXd &d_outputs);
    void train(
        Eigen::MatrixXd &training_sample_i,
        Eigen::MatrixXd &training_sample_o,
        unsigned int nbr_epoch,
        unsigned int batch_size,
        double learning_rate,
        Eigen::MatrixXd &eval_input,
        Eigen::MatrixXd &eval_output);
    void core_train();
    double evaluate(
        Eigen::MatrixXd &eval_input,
        Eigen::MatrixXd &eval_output,
        unsigned int epoch);
    void evaluate_MNIST(
        Eigen::MatrixXd &eval_input,
        Eigen::MatrixXd &eval_output,
        unsigned int epoch);

    sf::Thread train_thread;
    sf::Mutex chart_mutex;
    bool thread_state;
    Gnuplot graph;

  private:
    unsigned int _nbr_layer;
    unsigned int _batch_size;
    unsigned int _nbr_epoch;
    double _learning_rate;
    std::vector<unsigned int> _layers;
    std::vector<Eigen::MatrixXd> _weights;
    std::vector<Eigen::MatrixXd> _bias;
    std::vector<Eigen::MatrixXd> _zs;
    std::vector<Eigen::MatrixXd> _activations;
    std::vector<Eigen::MatrixXd> _errors;
    struct Data *_mnist_data;
    std::vector<double> _x;
    std::vector<double> _y;
};

#endif
