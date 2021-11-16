#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                // TODO
                m.data[i][j]=1/(1+exp(-x));
            } else if (a == RELU){
                // TODO
                if (x>=0){
                    m.data[i][j]=x;
                }
                else{
                    m.data[i][j]=0;
                }
            } else if (a == LRELU){
                // TODO
                if (x>=0){
                    m.data[i][j]=x;
                }
                else{
                    m.data[i][j]=0.1*x;
                }
            } else if (a == SOFTMAX){
                // TODO
                m.data[i][j]=exp(x);
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
            for (int k =0;k<m.cols;++k){
                m.data[i][k]=m.data[i][k]/sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            // TODO: multiply the correct element of d by the gradient
            if(a == LOGISTIC){
                // TODO
                double derive = x*(1-x);
                d.data[i][j]=d.data[i][j]*derive;
            } else if (a == RELU){
                // TODO
                if (x>0){
                    d.data[i][j]=d.data[i][j]*1;
                }
                else{
                    d.data[i][j]=d.data[i][j]*0;
                }
            } else if (a == LRELU){
                // TODO
                if (x>0){
                    d.data[i][j]=d.data[i][j]*1;
                }
                else{
                    d.data[i][j]=d.data[i][j]*0.1;
                }
            }
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{
    
    l->in = in;  // Save the input for backpropagation
    
    
    // TODO: fix this! multiply input by weights and apply activation function.
    matrix out = make_matrix(in.rows, l->w.cols);
    out = matrix_mult_matrix(in, l->w);
    activate_matrix(out,l->activation);
    
    
    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)
    gradient_matrix(l->out,l->activation,delta);
    
    
    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
    matrix dw = make_matrix(l->w.rows, l->w.cols); // replace this
    matrix xt = transpose_matrix(l->in);
    dw = matrix_mult_matrix(xt,delta);
    l->dw = dw;
    
    
    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
    matrix dx = make_matrix(l->in.rows, l->in.cols); // replace this
    matrix wt = transpose_matrix(l->w);
    dx = matrix_mult_matrix(delta,wt);
    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // save it to l->v
    matrix dwt = make_matrix(l->dw.rows, l->dw.cols);
    for (int i=0 ; i<l->dw.rows;i++){
        for (int j=0; j<l->dw.cols;j++){
            dwt.data[i][j]= l->dw.data[i][j] - decay*l->w.data[i][j] + momentum*l->v.data[i][j];
        }
    }
    free_matrix(l->v);
    l->v = dwt;
    
    // Update l->w
    matrix w = l->w;
    l->w = axpy_matrix(rate,dwt,w);
    
    // Remember to free any intermediate results to avoid memory leaks
    
}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions
//

// 5.2.2.1 Why might we be interested in both training accuracy and testing accuracy?
// What do these two numbers tell us about our current model?
// TODO
// Akurasi training menunjukkan seberapa baiknya model dapat mengenali gambar yang telah dilatih sebelumnya,
// sedangkan akurasi testing menunjukkan sebarapa baiknya model dapat mengenali gambar yang baru.

// 5.2.2.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
// TODO
// ==========================================================================
//      LR      |    ACC TRAINING    |    ACC TESTING     |    FINAL LOSS   |
// ==========================================================================
//      10      |    9.9%            |    10%             |    NAN          | (final loss di iterasi ke 6 19.430558)
//      1       |    87.57%          |    88.02%          |    0.479951     |
//      0.1     |    91.53%          |    91.11%          |    0.250382     |
//      0.01    |    90.37%          |    91%             |    0.290485     |
//      0.001   |    85.89%          |    86.81%          |    0.555315     |
// ==========================================================================

// 5.2.2.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
// TODO
// ========================================================
//      decay   |    ACC TRAINING    |    ACC TESTING     |
// ========================================================
//      1       |    89.78%          |    90.57%          |
//      0.1     |    90.30%          |    91.04%          |
//      0.01    |    90.36%          |    91%             |
//      0.001   |    90.37%          |    91%             |
//      0.0001  |    90.37%          |    91%             |
//      0.00001 |    90.37%          |    91%             |
// ========================================================


// 5.2.3.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
// TODO
// ========================================================================
//      ACTIVATION  |   ACC TRAINING  |    ACC TESTING   |   FINAL LOSS   |
// ========================================================================
//      LOGISTIC    |    89.0%        |    89.76%        |     0.414435   |
//      RELU        |    92.75%       |    92.97%        |     0.190136   |
//      LRELU       |    92.53%       |    92.75%        |     0.196135   |
// ========================================================================
// Activation logistic --> pengurangan loss agak lambat, hal ini ditandai dengan loss terakhir dari iterasi ke 1000 adalah 0.41
// Activation relu --> pengurangan loss lebih cepat dibandingkan logistic
// Activation lrelu --> tidak berbeda jauh performansinya dengan relu
// menurut akurasi, fungsi aktivasi yang terbaik adalah fungsi aktivasi relu



// 5.2.3.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
// TODO
// Fungsi aktivasi yang digunakan adalah RELU (terkait dengan poin 5.2.3.1)
// ========================================================
//      LR      |    ACC TRAINING    |    ACC TESTING     |
// ========================================================
//      10      |    9.91%           |    10.0%           |
//      1       |    20.45%          |    20.6%           |
//      0.1     |    95.88%          |    95.34%          |
//      0.01    |    92.75%          |    92.97%          |
//      0.001   |    86.88%          |    87.7%           |
// ========================================================
// LR terbaik pada model ini adalah 0.1 dengan akurasi training sebesar 95.88% dan akurasi testing sebesar 95.34%

// 5.2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
// TODO
// Fungsi aktivasi yang digunakan adalah relu dengan LR 0.1
// =======================================================================
//      decay   |    ACC TRAINING    |    ACC TESTING     |  FINAL LOSS  |
// =======================================================================
//      1       |    91.92%          |    91.99%          |  0.200494    |
//      0.1     |    95.94%          |    95.83%          |  0.084105    |
//      0.01    |    95.83%          |    95.43%          |  0.121643    |
//      0.001   |    95.64%          |    95.13%          |  0.124016    |
//      0.0001  |    95.90%          |    95.42%          |  0.118124    |
// =======================================================================
// Weight decay tidak berpengaruh secara signifikan pada akurasi model

// 5.2.3.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// TODO
// Fungsi aktivasi yang digunakan adalah relu dengan LR 0.1
// =======================================================================
//      decay   |    ACC TRAINING    |    ACC TESTING     |  FINAL LOSS  |
// =======================================================================
//      1       |    94.07%          |    94.23%          |  0.203054    |
//      0.1     |    97.19%          |    96.57%          |  0.059925    |
//      0.01    |    98.31%          |    97.10%          |  0.014535    |
//      0.001   |    98.21%          |    97.15%          |  0.025226    |
//      0.0001  |    98.45%          |    97.47%          |  0.046766    |
// =======================================================================


// 5.3.2.1 How well does your network perform on the CIFAR dataset?
// TODO
// LR : 0.01, Activation RELU, Iterasi 3000, momentum 0.9, decay .0001
// Akurasi training --> training 47.1%, testing 45.7%
// LR : 0.01, Activation RELU, Iterasi 1000, momentum 0.9, decay .0001
// Akurasi training --> training 41.0%, testing 40.6%
// LR : 0.001, Activation RELU, Iterasi 1000, momentum 0.9, decay .0001
// Akurasi training --> training 33.4%, testing 33.0%

// LR : 0.01, Activation RELU, Iterasi 4000, momentum 0.9, decay .0001
// Akurasi training --> training 48.13%, testing 46.3%





