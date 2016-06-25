/*

*/


#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <limits>

#ifndef MAT_H
#define MAT_H
#define PAUSE {printf("Press \"Enter\" to continue\n"); fflush(stdin); getchar(); fflush(stdin);}

#define MIN(x,y) ( (x) < (y) ? (x) : (y) )
#define MAX(x,y) ((x)>(y)?(x):(y))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))


typedef std::numeric_limits< double > dbl;


using namespace std;

void doubleCopy(double* matrix_old, double* matrix_new, int ns);

void doubleCopy(double* matrix_old, double* matrix_new, int start, int ns);

void doubleZeros(double* matrix_new, int ns);

void doubleZeros(double* matrix_new, int start, int ns);

void doubleAdd(double* matrix_new, double added, int ns);

void doubleAdd(double* matrix_new, double added, int start, int ns);

void doublePrint(double* p, string file, int rows, int cols);

void doublePrint(double* p, string file, int rows);

// Declarations
class matrix;
class imat;
double det(const matrix& a);
matrix diag(const int n);
matrix diag(const matrix& v);
matrix ones(const int rows, const int cols);
int size(const matrix& a, const int i);
matrix zeros(const int rows, const int cols);


double hp_autocorr(matrix mat, matrix matl);
matrix hp_autocorr_infl(matrix mat, matrix matl);
double central_moment(matrix mat, double order);
matrix central_moment_infl(matrix mat, double order);



/* MyException class */
struct MyException : public std::exception
{
   std::string s;
   MyException(std::string ss) : s(ss) {}
   ~MyException() throw () {} // Updated
   const char* what() const throw() { return s.c_str(); }
};

/* -----------------------
		Matrix class
------------------------*/


class matrix
{
public:
  // constructor
  matrix()
  {
	
    // create a matrix object without content
    p = NULL;
    rows = 0;
    cols = 0;
  }

  // constructor
  matrix(const int row_count, const int column_count)
  {
    // create a matrix object with given number of rows and columns
    p = NULL;

    /*if (row_count > 0 && column_count > 0)
    {*/
      rows = row_count;
      cols = column_count;

      p = new double[rows*cols];
      //p[0] = new double[rows*cols];
      for (int r = 0; r < rows; r++)
      {

        // initially fill in zeros for all values in the matrix;
        for (int c = 0; c < cols; c++)
        {
          p[r+c*rows] = 0;
        }
      }
    //}
  }
  
  matrix(const int row_count){
  	rows=row_count;
  	cols=1;
  	
  	p = new double[rows*cols];
  	//p[0] = new double[rows];
  	for (int r = 0; r < rows; r++)
    {
        // initially fill in zeros for all values in the matrix;
        p[r] = 0;
    }
  }
  
  // assignment operator
  matrix(const matrix& a)
  {
    rows = a.rows;
    cols = a.cols;
    p = new double[a.cols*a.rows];
    //p[0] = new double[a.cols*a.rows];
    for (int r = 0; r < a.rows; r++)
    {

      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
        p[r+c*rows] = a.p[r+c*rows];
      }
    }
  }

  matrix(double* q, const int row_count){
    rows = row_count;
    cols = 1;
    p = q;
  }

  matrix(double* q, const int row_count, const int col_count){
    rows = row_count;
    cols = col_count;
    p = q;
  }

  // index operator. You can use this class like mymatrix(col, row)
  // the indexes are one-based, not zero based.
  double& operator()(const int r, const int c)
  {
    /*if (p != NULL && r >= 0 && r <= rows-1 && c >= 0 && c <= cols-1)
    {*/
      return p[r+c*rows];
    //}
    /*else
    {
      throw MyException("Subscript out of range");
    }*/
  }

	double& operator()(int i){
		//if (p != NULL && i < rows*cols){
			/*if(cols==1){
				return p[i];
			}
			else if(rows==1){*/
				return p[i];
			/*}
			else
			{
			  throw MyException("Not a vector");
			}*/
		/*}
		else
		{
		  throw MyException("Subscript out of range");
		}*/
	}
  // index operator. You can use this class like mymatrix.get(col, row)
  // the indexes are one-based, not zero based.
  // use this function get if you want to read from a const matrix
  double get(const int r, const int c) const
  {
    /*if (p != NULL && r >= 0 && r <= rows-1 && c >= 0 && c <= cols-1)
    {*/
      return p[r+c*rows];
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }*/
  }
  
  matrix col(const int c)
  {
  	matrix col(rows, 1);
    /*if (p != NULL && c >= 0 && c <= cols-1)
    {*/
	  for(int i = 0; i < rows; i++){
	  	col.p[i]=p[i+c*rows];
	  }
      return col;
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }*/
  }
  
  matrix subcol(const int c, const int rl, const int rh){
    matrix col(rh-rl, 1);
    /*if (p != NULL && c >= 0 && c <= cols-1 && rl >= 0 && rh > rl && rh <= rows)
    {*/
	  for(int i = 0; i < rh-rl; i++){
	  	col.p[i]=p[i+rl+c*rows];
	  }
      return col;
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }  */
  }
 
  void reshape(int new_row, int new_col){
    rows = new_row;
    cols = new_col;
  }

  matrix subvec(const int rl, const int rh){
    matrix col(rh-rl);
    /*if (p != NULL && rl >= 0 && rh > rl && rh <= rows)
    {*/
	  for(int i = 0; i < rh-rl; i++){
	  	col.p[i]=p[i+rl];
	  }
      return col;
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }*/
  }
  
  matrix submatrow(const int rl, const int rh){
    matrix col(rh-rl,cols);
    /*if (p != NULL && rl >= 0 && rh > rl && rh <= rows)
    {*/
	for(int c = 0; c < cols; c++){
	  for(int i = 0; i < rh-rl; i++){
	  	col.p[i + c*(rh-rl)]=p[i+rl + c*rows];
	  }
	}
      return col;
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }*/
  }
	 
  double* point()
  {
  	return p;
  }
  
  void col_sub(matrix col, int c)
  {
    //if (p != NULL && c >= 0 && c <= cols-1 && col.cols * col.rows == rows){
        for(int i = 0; i < rows; i++){
            p[i+c*rows] = col.p[i];
        }
    /*}
    else{
        throw MyException("Error coleq");
    }*/
  }
  
  matrix row(const int r)
  {
  	matrix row(cols);
    //if (p != NULL && r >= 0 && r <= rows-1){
	  for(int i = 0; i < cols; i++){
	  	row.p[i]=p[r+rows*i];
	  }
      return row;
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }*/
  }

  void row_sub(matrix row, int r)
  {
    //if (p != NULL && c >= 0 && c <= cols-1 && col.cols * col.rows == rows){
        for(int i = 0; i < cols; i++){
            p[r+i*rows] = row.p[i];
        }
    /*}
    else{
        throw MyException("Error coleq");
    }*/
  }

  
  // assignment operator
  matrix& operator= (const matrix& a)
  {
    if(rows != a.rows || cols != a.cols || p == NULL){
        
        if(p != NULL){
            delete [] p;
        }
        
        rows = a.rows;
        cols = a.cols;
        
        p = new double[a.cols*a.rows];
    }
    
    //p[0] = new double[a.cols];
    for (int r = 0; r < a.rows; r++)
    {
      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
        p[r+c*rows] = a.p[r+c*rows];
      }
    }
    return *this;
  }

  // add a double value (elements wise)
  matrix& Add(const double v)
  {
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < cols; c++)
      {
        p[r+c*rows] += v;
      }
    }
     return *this;
  }

  // subtract a double value (elements wise)
  matrix& Subtract(const double v)
  {
    return Add(-v);
  }

  // multiply a double value (elements wise)
  matrix& Multiply(const double v)
  {
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < cols; c++)
      {
        p[r+c*rows] *= v;
      }
    }
     return *this;
  }

  // divide a double value (elements wise)
  matrix& Divide(const double v)
  {
     return Multiply(1/v);
  }

  // addition of matrix with matrix
  friend matrix operator+(const matrix& a, const matrix& b)
  {
    // check if the dimensions match
    //if (a.rows == b.rows && a.cols == b.cols){
      matrix res(a.rows, a.cols);

      for (int r = 0; r < a.rows; r++)
      {
        for (int c = 0; c < a.cols; c++)
        {
          res.p[r+c*a.rows] = a.p[r+c*a.rows] + b.p[r+c*a.rows];
        }
      }
      return res;
    /*}
    else
    {
      // give an error
      throw MyException("Dimensions does not match");
    }*/

    // return an empty matrix (this never happens but just for safety)
    return matrix();
  }

  // addition of matrix with matrix
  friend matrix operator+=(const matrix& a, const matrix& b)
  {
    // check if the dimensions match
    //if (a.rows == b.rows && a.cols == b.cols){
      matrix res(a.rows, a.cols);

      for (int r = 0; r < a.rows; r++)
      {
        for (int c = 0; c < a.cols; c++)
        {
          a.p[r+c*a.rows] = a.p[r+c*a.rows] + b.p[r+c*a.rows];
        }
      }
    /*}
    else
    {
      // give an error
      throw MyException("Dimensions does not match");
    }*/

    // return an empty matrix (this never happens but just for safety)
    return matrix();
  }
  
  // addition of matrix with double
  friend matrix operator+ (const matrix& a, const double b)
  {
    matrix res = a;
    res.Add(b);
    return res;
  }
  // addition of double with matrix
  friend matrix operator+ (const double b, const matrix& a)
  {
    matrix res = a;
    res.Add(b);
    return res;
  }

  // subtraction of matrix with matrix
  friend matrix operator- (const matrix& a, const matrix& b)
  {
    // check if the dimensions match
    //if (a.rows == b.rows && a.cols == b.cols){
      matrix res(a.rows, a.cols);

      for (int r = 0; r < a.rows; r++)
      {
        for (int c = 0; c < a.cols; c++)
        {
          res.p[r+c*a.rows] = a.p[r+c*a.rows] - b.p[r+c*a.rows];
        }
      }
      return res;
    /*}
    else
    {
      // give an error
      throw MyException("Dimensions does not match");
    }*/

    // return an empty matrix (this never happens but just for safety)
    return matrix();
  }

  // subtraction of matrix with double
  friend matrix operator- (const matrix& a, const double b)
  {
    matrix res = a;
    res.Subtract(b);
    return res;
  }
  // subtraction of double with matrix
  friend matrix operator- (const double b, const matrix& a)
  {
    matrix res = -a;
    res.Add(b);
    return res;
  }

  // operator unary minus
  friend matrix operator- (const matrix& a)
  {
    matrix res(a.rows, a.cols);

    for (int r = 0; r < a.rows; r++)
    {
      for (int c = 0; c < a.cols; c++)
      {
        res.p[r+c*a.rows] = -a.p[r+c*a.rows];
      }
    }

    return res;
  }

  // operator multiplication
  friend matrix operator* (const matrix& a, const matrix& b)
  {
    // check if the dimensions match
    //if (a.cols == b.rows){
      matrix res(a.rows, b.cols);

      for (int r = 0; r < a.rows; r++)
      {
        for (int c = 0; c < b.cols; c++)
        {
          for (int c_res = 0; c_res < a.cols; c_res++)
          {
            res.p[r+c*a.rows] += a.p[r+c_res*a.rows] * b.p[c_res+c*b.rows];
          }
        }
      }
      return res;
    /*}
    else
    {
      // give an error
      throw MyException("Dimensions does not match");
    }*/

    // return an empty matrix (this never happens but just for safety)
    return matrix();
  }

  // multiplication of matrix with double
  friend matrix operator* (const matrix& a, const double b)
  {
    matrix res = a;
    res.Multiply(b);
    return res;
  }
  // multiplication of double with matrix
  friend matrix operator* (const double b, const matrix& a)
  {
    matrix res = a;
    res.Multiply(b);
    return res;
  }

  // division of matrix with matrix
  friend matrix operator/ (const matrix& a, const matrix& b)
  {
    matrix res(a.rows, a.cols);

    for (int r = 0; r < a.rows; r++)
    {
      for (int c = 0; c < a.cols; c++)
      {
        res.p[r+c*a.rows] = a.p[r+c*a.rows] / b.p[r+c*a.rows];
      }
    }

    return res;
  }

  // division of matrix with double
  friend matrix operator/ (const matrix& a, const double b)
  {
    matrix res = a;
    res.Divide(b);
    return res;
  }

  // division of double with matrix
  friend matrix operator/ (const double b, const matrix& a)
  {
    matrix b_matrix(0, 0);
    b_matrix(0,0) = b;

    matrix res = b_matrix / a;
    return res;
  }

  // division of double with matrix
  friend matrix operator & (const matrix& a, const matrix& b)
  {
    matrix res(a.rows, a.cols);

    for (int r = 0; r < a.rows; r++)
    {
      for (int c = 0; c < a.cols; c++)
      {
        res.p[r+c*a.rows] = a.p[r+c*a.rows] * b.p[r+c*a.rows];
      }
    }

    return res;
  }


  /**
   * returns the minor from the given matrix where
   * the selected row and column are removed
   */
  matrix Minor(const int row, const int col) const
  {
    matrix res;
    //if (row >= 0 && row < rows && col >= 0 && col < cols){
      res = matrix(rows - 1, cols - 1);
      
      // copy the content of the matrix to the minor, except the selected
      for (int r = 0; r < (rows - (row+1 >= rows)); r++)
      {
        for (int c = 0; c < (cols - (col+1 >= cols)); c++)
        {
          res(r - (r > row), c - (c > col)) = p[r+c*rows];
        }
      }
      
    /*}
    else
    {
      printf("Index for minor out of range");
      throw MyException("Index for minor out of range");
    }*/

    return res;
  }

  /*
   * returns the size of the i-th dimension of the matrix.
   * i.e. for i=1 the function returns the number of rows,
   * and for i=2 the function returns the number of columns
   * else the function returns 0
   */
  int size(const int i) const
  {
    if (i == 1)
    {
      return rows;
    }
    else if (i == 2)
    {
      return cols;
    }
    return 0;
  }

  // returns the number of rows
  int nrows() const
  {
    return rows;
  }

  // returns the number of columns
  int ncols() const
  {
    return cols;
  }
	
  void randu(){
		for(int i = 0; i < rows; i++){
	  		for(int j = 0; j < cols; j++){
				p[i+rows*j]=((double) rand() / (double) (RAND_MAX));
				}
			}
	}
	
  void zeros(){
		for(int i = 0; i < rows; i++){
	  		for(int j = 0; j < cols; j++){
				p[i+rows*j]=0;
				}
			}
	}
	
double determinant(int k)
{
  double s=1,det=0;
  matrix b(rows, cols);
  int i,j,m,n,c;
  if (k==1)
    {
     return (p[0]);
    }
  else
    {
     det=0;
     for (c=0;c<k;c++)
       {
        m=0;
        n=0;
        for (i=0;i<k;i++)
          {
            for (j=0;j<k;j++)
              {
                b.p[j*rows+i]=0;
                if (i != 0 && j != c)
                 {
                   b.p[m*rows+n]=p[j*rows+i];
                   if (n<(k-2))
                    n++;
                   else
                    {
                     n=0;
                     m++;
                     }
                   }
               }
             }
          det=det + s * (p[c*rows] * b.determinant(k-1));
          s=-1 * s;
          }
    }
 
    return (det);
}
		
  matrix t() const
  {
  	matrix T(cols,rows);
  	for(int i = 0; i < rows; i++){
  		for(int j = 0; j < cols; j++){
  			T(j,i) = p[i+rows*j];
  		}
  	}
    return T;
  }

  matrix floorzero() const
  {
    matrix P(rows,cols);
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        if(p[i+rows*j] > 0){
          P(i,j) = p[i+rows*j];
        }
      }
    }
    return P;
  }

  matrix geq() const
  {
    matrix P(rows,cols);
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++){
        if(p[i+rows*j] > 0){
          P(i,j) = 1.0;
        }
      }
    }
    return P;
  }

  // print the contents of the matrix
  void print() const
  {
    if (p != NULL)
    {
      printf("[");
      for (int r = 0; r < rows; r++)
      {
        if (r > 0)
        {
          printf(" ");
        }
        for (int c = 0; c < cols-1; c++)
        {
          printf("%.3f, ", p[r+rows*c]);
        }
        if (r < rows-1)
        {
          printf("%.3f;\n", p[r+rows*(cols-1)]);
        }
        else
        {
          printf("%.3f]\n", p[r+rows*(cols-1)]);
        }
      }
    }
    else
    {
      // matrix is empty
      printf("[ ]\n");
    }
  }

  void print(string file){
    ofstream simfile;
    simfile.open(file.c_str());
    simfile.precision(dbl::digits10);
    for(int i = 0; i < rows; i++){
      simfile << p[i];
      for(int j = 1; j < cols; j++){
        simfile << ", " << p[i+rows*j];
      }
      simfile << endl;
    }
    
    simfile.close();
  }

  // print the contents of the matrix
  void printv() const
  {
    if (p != NULL)
    {
      printf("[");
      for (int r = 0; r < rows; r++)
      {
        if (r > 0)
        {
          printf(" ");
        }
        for (int c = 0; c < cols-1; c++)
        {
          printf("%.3f, ", p[r+rows*c]);
        }
        if (r < rows-1)
        {
          printf("%.3f; ", p[r+rows*(cols-1)]);
        }
        else
        {
          printf("%.3f]", p[r+rows*(cols-1)]);
        }
      }
    }
    else
    {
      // matrix is empty
      printf("[ ]\n");
    }
  }

  matrix cluster(){
     double vi;
     matrix clust(cols, cols);
     for(int a = 0; a < cols; a++){
         for(int b = a; b < cols; b++){
	    vi = 0;
	    for(int i = 0; i < rows; i++){
	        vi += p[i + a*cols] * p[i + a*cols];
		for(int j = i + 1; j < rows; j++){
		    vi += 2*p[i + a*cols] * p[j + b*cols];
		}
    	    }
            clust(a, b) = vi;
            clust(b, a) = vi;
          }
      }
      return clust;
  }

//public:
  // destructor
  virtual ~matrix()
  {
    // clean up allocated memory
    //printf("Pre delete ");
    //cout << "rows: " << rows << " cols: " << cols << endl;
    delete [] p;
    //printf("Object deleted \n");
    p = NULL;
    //printf(" Object cleared \n");*/
  }

private:
  int rows;
  int cols;
  double* p;     // pointer to a matrix with doubles
};

int svd(matrix &mat_arg_a, matrix &mat_arg_w, matrix &mat_arg_v);

int size(const matrix& a, const int i);

matrix ones(const int rows, const int cols);

matrix eye(int rows);

//matrix inverse(matrix A);

matrix zeros(const int rows, const int cols);

double mean(matrix mat);

double freq(matrix mat);

matrix cmean(matrix mat);

matrix csum(matrix mat);

matrix rsum(matrix mat);
matrix pinv(matrix a);
matrix pinv(double *a, int m, int n);

double var(matrix mat);
double sum(matrix mat);
matrix mabs(matrix mat);
matrix cmeansq(matrix mat);
matrix csqrtmeansq(matrix mat);
matrix sqrt(matrix mat);

matrix cov(matrix mat);
double cov(matrix mat1, matrix mat2);


matrix diag(const int n);

matrix diag(const matrix& v);

matrix sqrtdiags(const matrix& v);

double det(const matrix& a);

void Swap(double& a, double& b);

matrix inv(matrix a);

matrix chol(matrix A);

matrix sub_f(matrix L, matrix b);

matrix sub_b(matrix U, matrix b);

matrix solve_sym(matrix S, matrix b);

void dsvd(matrix mat, matrix &a, matrix &w, matrix &v);

/* -----------------------
		Integer matrix class
------------------------*/


class imat
{
public:
  // constructor
  imat()
  {
	
    // create a matrix object without content
    p = NULL;
    rows = 0;
    cols = 0;
  }

  // constructor
  imat(const int row_count, const int column_count)
  {
    // create a matrix object with given number of rows and columns
    p = NULL;

    if (row_count > 0 && column_count > 0)
    {
      rows = row_count;
      cols = column_count;

      p = new int[rows*cols];
      for (int r = 0; r < rows; r++)
      {
        // initially fill in zeros for all values in the matrix;
        for (int c = 0; c < cols; c++)
        {
          p[r+c*rows] = 0;
        }
      }
    }
  }
  // constructor
  imat(const int row_count){
  	rows=row_count;
  	cols=1;
  	
  	p = new int[rows*cols];
  	
  	for (int r = 0; r < rows; r++)
    {
        // initially fill in zeros for all values in the matrix;
        p[r] = 0;
    }
  }
  
  // assignment operator
  imat(const imat& a)
  {
    rows = a.rows;
    cols = a.cols;
    p = new int[a.rows*a.cols];
    for (int r = 0; r < a.rows; r++)
    {
      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
        p[r+a.rows*c] = a.p[r+a.rows*c];
      }
    }
  }

  // index operator. You can use this class like mymatrix(col, row)
  // the indexes are one-based, not zero based.
  int& operator()(const int r, const int c)
  {
    //if (p != NULL && r >= 0 && r <= rows-1 && c >= 0 && c <= cols-1){
      return p[r+rows*c];
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }*/
  }

	int& operator()(int i){
		//if (p != NULL && i < rows*cols){
			//if(cols==1){
				return p[i];
			/*}
			else if(rows==1){
				return p[i];
			}
			else
			{
			  throw MyException("Not a vector");
			}*/
		/*}
		else
		{
		  throw MyException("Subscript out of range");
		}*/
	}
  // index operator. You can use this class like mymatrix.get(col, row)
  // the indexes are one-based, not zero based.
  // use this function get if you want to read from a const matrix
  int get(const int r, const int c) const
  {
    //if (p != NULL && r >= 0 && r <= rows-1 && c >= 0 && c <= cols-1){
      return p[r+rows*c];
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }*/
  }
  
  imat col(const int c)
  {
  	imat col(rows, 1);
    //if (p != NULL && c >= 0 && c <= cols-1){
	  for(int i = 0; i < rows; i++){
	  	col.p[i]=p[i+rows*c];
	  }
      return col;
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }*/
  }
  
  imat subcol(const int c, const int rl, const int rh){
    imat col(rh-rl, 1);
    //if (p != NULL && c >= 0 && c <= cols-1 && rl >= 0 && rh > rl && rh <= rows){
	  for(int i = 0; i < rh-rl; i++){
	  	col.p[i]=p[i+rl+c*rows];
	  }
      return col;
    /*}
    else
    {
      throw MyException("Subscript out of range");
    }*/
  }
  
  imat subvec(const int rl, const int rh){
    imat col(rh-rl);
    if (p != NULL && rl >= 0 && rh > rl && rh <= rows)
    {
	  for(int i = 0; i < rh-rl; i++){
	  	col.p[i]=p[i+rl];
	  }
      return col;
    }
    else
    {
      throw MyException("Subscript out of range");
    }
  }
  
  imat row(const int r)
  {
  	imat row(1, cols);
    if (p != NULL && r >= 0 && r <= rows-1)
    {
	  for(int i = 0; i < cols; i++){
	  	row.p[i]=p[r+rows*i];
	  }
      return row;
    }
    else
    {
      throw MyException("Subscript out of range");
    }
  }
  
    // assignment operator
  imat& operator= (const imat& a)
  {
    rows = a.rows;
    cols = a.cols;
    
    delete [] p;
    
    p = new int[a.rows*a.cols];
    for (int r = 0; r < a.rows; r++)
    {
      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
        p[r+a.rows*c] = a.p[r+a.rows*c];
      }
    }
    return *this;
  }

  // add a double value (elements wise)
  imat& Add(const int v)
  {
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < cols; c++)
      {
        p[r+rows*c] += v;
      }
    }
     return *this;
  }

  // subtract a double value (elements wise)
  imat& Subtract(const int v)
  {
    return Add(-v);
  }

  // multiply a double value (elements wise)
  imat& Multiply(const int v)
  {
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < cols; c++)
      {
        p[r+rows*c] *= v;
      }
    }
     return *this;
  }
	
  // addition of matrix with double
  friend imat operator+ (const imat& a, const int b)
  {
    imat res = a;
    res.Add(b);
    return res;
  }
  // addition of double with matrix
  friend imat operator+ (const int b, const imat& a)
  {
    imat res = a;
    res.Add(b);
    return res;
  }
  
  // operator unary minus
  friend imat operator- (const imat& a)
  {
    imat res(a.rows, a.cols);

    for (int r = 0; r < a.rows; r++)
    {
      for (int c = 0; c < a.cols; c++)
      {
        res.p[r+a.rows*c] = -a.p[r+a.rows*c];
      }
    }

    return res;
  }
  
  // subtraction of matrix with double
  friend imat operator- (const imat& a, const int b)
  {
    imat res = a;
    res.Subtract(b);
    return res;
  }
  
  // subtraction of double with matrix
  friend imat operator- (const int  b, const imat& a)
  {
    imat res = -a;
    res.Add(b);
    return res;
  }

  int size(const int i) const
  {
    if (i == 1)
    {
      return rows;
    }
    else if (i == 2)
    {
      return cols;
    }
    return 0;
  }

  // returns the number of rows
  int nrows() const
  {
    return rows;
  }

  // returns the number of columns
  int ncols() const
  {
    return cols;
  }

  imat t() const
  {
  	imat T(cols,rows);
  	for(int i = 0; i < cols; i++){
  		for(int j = 0; j < rows; j++){
  			T(j,i) = p[i+rows*j];
  		}
  	}
    return T;
  }
  
  // Return pointer
  int* point()
  {
  	return p;
  }
  
  // print the contents of the matrix
void print() const
  {
    if (p != NULL)
    {
      printf("[");
      for (int r = 0; r < rows; r++)
      {
        if (r > 0)
        {
          printf(" ");
        }
        for (int c = 0; c < cols-1; c++)
        {
          printf("%i, ", p[r+rows*c]);
        }
        if (r < rows-1)
        {
          printf("%i;\n", p[r+rows*(cols-1)]);
        }
        else
        {
          printf("%i]\n", p[r+rows*(cols-1)]);
        }
      }
    }
    else
    {
      // matrix is empty
      printf("[ ]\n");
    }
  }

  void print(string file){
    ofstream simfile;
    simfile.open(file.c_str());
    
    for(int i = 0; i < rows; i++){
      simfile << p[i];
      for(int j = 1; j < cols; j++){
        simfile << ", " << p[i+rows*j];
      }
      simfile << endl;
    }
    
    simfile.close();
  }

public:
  // destructor
  ~imat()
  {
    // clean up allocated memory
    
    delete [] p;
    p = NULL;
  }

private:
  int rows;
  int cols;
  int* p;     // pointer to a matrix with doubles
};

























class matrix3
{
public:
  // constructor
  matrix3()
  {
	
    // create a matrix object without content
    p = NULL;
    rows = 0;
    cols = 0;
    spans = 0;
  }

  // constructor
  matrix3(const int row_count, const int column_count, const int span_count)
  {
    // create a matrix object with given number of rows and columns
    p = NULL;

    if (row_count > 0 && column_count > 0 && span_count > 0)
    {
      rows = row_count;
      cols = column_count;
      spans = span_count;

      p = new double[rows*cols*spans];
      //p[0] = new double[rows*cols];
      for (int r = 0; r < rows; r++)
      {

        // initially fill in zeros for all values in the matrix;
        for (int c = 0; c < cols; c++)
        {
            for(int d = 0; d < spans; d++){
                p[r+c*rows+d*rows*cols] = 0;
            }
        }
      }
    }
  }
  
  // assignment operator
  matrix3(const matrix3& a)
  {
    rows = a.rows;
    cols = a.cols;
    spans = a.spans;
    p = new double[a.cols*a.rows*a.spans];
    //p[0] = new double[a.cols*a.rows];
    for (int r = 0; r < a.rows; r++)
    {

      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
        for(int d = 0; d < a.spans; d++){
            p[r+c*rows+d*rows*cols] = a.p[r+c*rows+d*rows*cols];
        }
      }
    }
  }

  // assignment operator
  matrix3& operator= (const matrix3& a)
  {
    if(rows != a.rows || cols != a.cols || spans != a.spans || p == NULL){
        
        if(p != NULL){
            delete [] p;
        }
        
        rows = a.rows;
        cols = a.cols;
        spans = a.spans;
        
        p = new double[a.cols*a.rows*a.spans];
    }
    
    //p[0] = new double[a.cols];
    for (int r = 0; r < a.rows; r++)
    {
      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
            for (int d = 0; d < a.spans; d++)
            {
                p[r+c*rows+d*rows*cols] = a.p[r+c*rows+d*rows*cols];
            }
      }
    }
    return *this;
  }
  
  // index operator. You can use this class like mymatrix(col, row)
  // the indexes are one-based, not zero based.
  double& operator()(const int r, const int c, const int d)
  {
    if (p != NULL && r >= 0 && r <= rows-1 && c >= 0 && c <= cols-1 && d >= 0 && d <= spans-1)
    {
      return p[r+c*rows+d*rows*cols];
    }
    else
    {
      throw MyException("Subscript out of range");
    }
  }
  
  // index operator. You can use this class like mymatrix.get(col, row)
  // the indexes are one-based, not zero based.
  // use this function get if you want to read from a const matrix
  double get(const int r, const int c, const int d) const
  {
    if (p != NULL && r >= 0 && r <= rows-1 && c >= 0 && c <= cols-1 && d >= 0 && d <= spans - 1)
    {
      return p[r+c*rows+d*rows*cols];
    }
    else
    {
      throw MyException("Subscript out of range");
    }
  }
  
  matrix col(const int c, const int d)
  {
  	matrix col(rows, 1);
    if (p != NULL && c >= 0 && c <= cols-1 && d >=0 && d <= spans-1)
    {
	  for(int i = 0; i < rows; i++){
	  	col(i)=p[i+c*rows+d*cols*rows];
	  }
      return col;
    }
    else
    {
      throw MyException("Subscript out of range (column)");
    }
  }
//public:
  // destructor
  virtual ~matrix3()
  {
    // clean up allocated memory
    cout << cols << " " << rows << endl;
    
    
    delete [] p;
    //printf("Object deleted");
    p = NULL;
  }

private:
  int rows;
  int cols;
  int spans;
  double* p;     // pointer to a matrix with doubles
};

matrix exp(matrix mat);

matrix pseudoinv(matrix a);
void svdcmp(matrix &a, matrix &w, matrix &v);
matrix log(matrix mat);

void jacobi_eigenvalue (matrix a, int it_max, matrix &v,
  matrix &d, int &it_num, int &rot_num );

matrix readcsv(string files, int rows, int cols);

struct ord{
	matrix vect;
	imat index;
};

void quickOrderi(double* p1, int* p2, int left, int right);
ord quickOrder(matrix arrr, imat orders, int left, int right);
void quickSort(double* p1, int left, int right);
#endif
