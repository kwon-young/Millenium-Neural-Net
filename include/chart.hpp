#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>
#include <cstdio> // popen
#include <cstring> // memset
#include <iostream>

#include "plot.h"

class Chart : public sf::Drawable
{
  public:
    Chart(const std::string& name, const sf::Vector2i& location);

    void update(double value);

    void draw(sf::RenderTarget& target, sf::RenderStates states) const;

  private:
    sf::plot::Plot plot_;
    std::string name_;
};

