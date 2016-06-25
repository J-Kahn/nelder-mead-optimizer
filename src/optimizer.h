#include <ctime>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
using namespace std;

double deviation(matrix a, matrix b){
  double dev = 0;
  for(int i = 0; i < a.nrows(); i++){
    dev += (a(i) - b(i)) * (a(i) - b(i));
  }
  return dev;
}

class NelderMeadOptimizer {
    public:
      
      double alpha, gamma, rho, sigma;
        matrix vectors;
        matrix values;
        matrix cog;
        
      NelderMeadOptimizer(int dimension, int npar, double termination_distance=0.001) {
          this->dimension = dimension;
          srand(time(NULL));
          alpha = 1;
          gamma = 2;
          rho = -0.5;
          sigma = 0.5;
          feed = 0;
          matrix vectors(dimension + 1, npar);
          this->termination_distance = termination_distance;
      }
      
      
      // termination criteria: each pair of vectors in the simplex has to
      // have a distance of at most `termination_distance`
      bool done() {
          if (vectors.nrows() < dimension) {
              return false;
          }
          for (int i=0; i<dimension+1; i++) {
              for (int j=0; j<dimension+1; j++) {
                  if (i==j) continue;
                  if (deviation(vector.row(i), vector.row(j)) > termination_distance) {
                      return false;
                  }
              }
          }
          return true;
      }
      
      void insert(matrix vec) {
          if (feed < dimension+1) {
              for(int j = 0; j < vectors.ncols(); j++){
                vectors(feed, j) = vec(j);
              }
              
              values(feed) = val;
              feed += 1;
          }
      }
      

      
      
    matrix reflect(){

      sort(vectors.begin(), vectors.end(), *this);
      
      for (int i = 1; i<=dimension; i++) {
          cog = cog + vectors.row(i);
      }
      cog /= dimension;

      matrix reflected = cog + (cog - vectors.row(0)) * alpha;
      
    }
        
    matrix step(matrix vec, double score) {
            
      matrix returned;
      
      if(feed < dimension + 1){
        
        returned = vectors.row(feed);

        feed++;
        
        if(feed < dimension){
          returned = vectors.row(feed);
        }
        else{
          returned = reflect();
        }

      }
      else{
        matrix reflected = reflect();

        if(deviation(reflected, vec) < 1e-12){
          
          if (score > values(1) && score < values(dimension)) {
              vectors.row_sub(reflected, 0);
              values(0) = score;
              returned = reflect();
          } else if (score > values(dimension)) {
              vectors.row_sub(reflected, 0);
              values(0) = score;
              returned = cog + (cog - vectors.row(0))*gamma;
              vectors.row_sub(reflected, 0);

            } else {
              returned = cog + (cog - vectors.row(0))*rho;
            }
        }
        else{
          if(score > values(0) ){
            values(0) = score;
            vectors.row_sub(vec, 0);
            returned = reflected;
          } else{
            
            if(values(0) > values(1)){
                returned = reflected;
            } else {
                feed = 1;
                
                for (int i=0; i<dimension; i++) {
                    vectors.row_sub(vectors.row(dimension) + (vectors.row(i) - vectors.row(dimension))*sigma, i);
                }
                
                temp = vectors.row(0);
                vectors.row_sub(vectors.row(dimension), temp);
                values(0) = values(dimension);
                returned = vectors.row(1);
                
            }
            
          }
        }
      }
      
      return returned;
    }

 
    private:
        int dimension;
        double termination_distance;

};
