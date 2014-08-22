#include <ovxx/library.hpp>
#include <ovxx/timer.hpp>

int main(int, char **)
{
  ovxx::library lib;
  ovxx::timer t;
  double delta = t.elapsed();
  t.restart();
  delta = t.elapsed();  
}
