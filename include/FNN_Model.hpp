
#ifndef FNN_MODEL_HPP
#define FNN_MODEL_HPP

#include <vector>
#include <Eigen/Dense>

class FNN_Model
{
  public:
    FNN_Model(
        std::vector<unsigned int> layers,
        unsigned int nbr_epoch,
        unsigned int batch_size=1);
    ~FNN_Model();

    void print_FNN();
    void Init();
    void ResizeBatch();
    void SetInput(Eigen::MatrixXd inputs);
    static double sigmoid(double input);
    void FeedForward();

  private:
    unsigned int _nbr_layer;
    unsigned int _batch_size;
    unsigned int _nbr_epoch;
    std::vector<unsigned int> _layers;
    std::vector<Eigen::MatrixXd> _weights;
    std::vector<Eigen::MatrixXd> _bias;
    std::vector<Eigen::MatrixXd> _zs;
    std::vector<Eigen::MatrixXd> _activations;
    std::vector<Eigen::MatrixXd> _errors;
};

#endif
