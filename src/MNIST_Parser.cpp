
#include "MNIST_Parser.hpp"
#include <iostream>
#include <stdio.h>
#include <bitset>

MNIST_Parser::MNIST_Parser (std::string data_directory):
  _data_directory(data_directory),
  _train_img(),
  _train_label(),
  _eval_img(),
  _eval_label()
{
  std::string train_img_f(_data_directory);
  std::string train_label_f(_data_directory);
  train_img_f += "/train-images.idx3-ubyte";
  train_label_f += "/train_labels.idx3-ubyte";
  _train_img.open(train_img_f.c_str(), std::ios::binary);
  _train_label.open(train_label_f.c_str(), std::ios::binary);
}

MNIST_Parser::~MNIST_Parser ()
{
  if (_train_img.is_open())
    _train_img.close();
  if (_train_label.is_open())
    _train_label.close();
}

void MNIST_Parser::read_MNIST_format(
    std::ifstream &file,
    Eigen::MatrixXd &storage,
    unsigned int header_size)
{
  unsigned char *buffer = new unsigned char [header_size];
  int *header = new int [header_size];
  for (int i = 0; i < 4; ++i) {
    file.read((char*)buffer, 4);
    //convert big-endian to little-endian
    header[i] = buffer[0] << 24 | buffer[1] << 16 | buffer[2] << 8 | buffer[3] << 0;
  }
  int img_size = header[2]*header[3];
  storage.resize(header[2]*header[3], header[1]);
  char *img = new char [img_size];
  for (int i = 0; i < header[1]; ++i) {
    file.read(img, img_size);
    for (int j = 0; j < img_size; ++j) {
      storage.col(i).row(j) << (double)(img[j]);
    }
  }
  delete[] buffer;
  delete[] header;
  delete[] img;
}

void MNIST_Parser::read_train_img (Eigen::MatrixXd &train_images)
{
  read_MNIST_format(_train_img, train_images, 4);
}

//void MNIST_Parser::read_train_label (Eigen::MatrixXd &train_labels)
//{
  //read_MNIST_format(_train_img, train_labels, 2);
//}

//void MNIST_Parser::read_eval_img (Eigen::MatrixXd &eval_images);
//void MNIST_Parser::read_eval_label (Eigen::MatrixXd &eval_labels);
