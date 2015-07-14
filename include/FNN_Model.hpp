
#ifndef FNN_MODEL_HPP
#define FNN_MODEL_HPP

#include <vector>
#include <list>
#include <Eigen/Dense>

class FNN_Model
{
  public:
    FNN_Model(
        std::vector<unsigned int> layers);
    ~FNN_Model();

    void print_FNN();
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
        double learning_rate);

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
};

#endif
