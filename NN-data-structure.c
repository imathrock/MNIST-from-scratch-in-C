// This code defines the data structures of each element present in a neural network
// It will contain structs that define neurons, layers of neurons, matricies for weights
// Matricies for biases and some linear algebra functions.

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>

#include"idx-file-parser.h"

// Neuron is something that holds a value.
// It need to have decimal precison, hence float is chosen.
struct neuron {
    float activation;
};

// Neuron layer consists of N neurons that is specified by constructor
// So a neuron layer can de described using an array of neurons, hence the following struct.
struct neuron_layer {
    struct neuron*N;
    int num_neurons; 
};

// Weight matrix is a 2d matrix that consists of, 
// all the weight values that connect 2 layers. Will need a weight matrix constructor.
struct weight_matrix {
    int rows;
    int cols;
    float**weights;
};

/*
A matrix that biases the neuron's activation value, This is subtracted from the 
product of weight matrix and neuron layer matrix to give the activation values of 
the next matrix.
*/
struct bias_matrix{
    int size;
    float*bias;
};

/// @brief constructs a neuron layer with each neuron containing a random number between 0 and 1 
struct neuron_layer* layer_constructor(int num_neurons){
    struct neuron_layer*layer = (struct neuron_layer*)malloc(sizeof(struct neuron_layer));
    layer->num_neurons = num_neurons;
    layer->N = (struct neuron*)malloc(num_neurons*sizeof(struct neuron));
    if(layer->N == NULL){printf("Failed to Allocate memory"); exit(1);}
    for(int i = 0; i < num_neurons; i++){
        layer->N[i].activation = ((float)rand() / RAND_MAX) - 0.5;
    }
    return layer;
}

/// @brief Constructs a matrix of weights with randomly assigned values
struct weight_matrix* weight_matrix_constructor(int rows, int cols){
    struct weight_matrix*W = (struct weight_matrix*)malloc(sizeof(struct weight_matrix));
    W->rows = rows;
    W->cols = cols;

    W->weights = (float**)malloc(rows*sizeof(float*));
    if(W->weights == NULL){printf("Failed to Allocate memory"); exit(1);}
    for(int i = 0; i < rows; i++){
        W->weights[i] = (float*)malloc(cols*sizeof(float));
        if(W->weights[i] == NULL){printf("Failed to Allocate memory"); exit(1);}
    }

    for(int i = 0; i< rows; i++){
        for(int j = 0; j < cols; j++){
            W->weights[i][j] = ((float)rand() / RAND_MAX) - 0.5;
        }
    }
    return W;
}

/// @brief Constructs a bias matrix with random values
struct bias_matrix* bias_matrix_constructor(int cols){
    struct bias_matrix*B = (struct bias_matrix*)malloc(sizeof(struct bias_matrix));
    B->size = cols;
    B->bias = (float*)malloc(cols*sizeof(float));
    if(B->bias == NULL){printf("Failed to Allocate memory"); exit(1);}
    for(int i = 0; i < cols; i++){
        B->bias[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
    return B;
}

/// @brief Frees the weight matrix
/// @param weight_matrix 
void destroy_weight_matrix(struct weight_matrix* weight_matrix) {
    if (weight_matrix == NULL) {
        return; // Nothing to do if the pointer is NULL
    }
    // Free each row of weights
    for (int i = 0; i < weight_matrix->rows; i++) {
        if (weight_matrix->weights[i] != NULL) {
            free(weight_matrix->weights[i]);
            weight_matrix->weights[i] = NULL; // Prevent dangling pointer
        }
    }
    // Free the weights pointer array
    if (weight_matrix->weights != NULL) {
        free(weight_matrix->weights);
        weight_matrix->weights = NULL; // Prevent dangling pointer
    }
}


/// @brief destroys and frees space for the neuron layer
/// @param layer 
void destroy_layer(struct neuron_layer* layer){
    if (layer->N != NULL) {
        free(layer->N);
        layer->N = NULL;
    }
    layer->num_neurons = 0;
}

/// @brief To be used for add biases from the layer's activation values
/// @param a Neuron Layer
/// @param b Bias Matrix
void add_matricies(struct neuron_layer *a, struct bias_matrix *b){
    if(a->num_neurons != b->size)
    {printf("matricies are incorrect in size, check again"); exit(1);}
    int size = a->num_neurons;
    for(int i = 0; i < size; i++){
        a->N[i].activation += b->bias[i];
    }
}

/// @brief Executes a dot product operation between the weight matrix and neuron layer
/// @param a initial neuron layer
/// @param w weight matrix
/// @param result resulting neuron layer, to be passed so values can be modified.
void matrix_dot_product(struct neuron_layer *a, struct weight_matrix *w, struct neuron_layer*result){
    if(a->num_neurons != w->cols)
    {printf("matricies are incorrect in size, check again"); exit(1);}
    int row = w->rows;
    for(int i = 0; i< row; i++){
        float prodsum = 0.0f;
        for(int j = 0; j< w->cols; j++){
            prodsum += (a->N[j].activation * w->weights[i][j]);
        }
        result->N[i].activation = prodsum;
    }
}

/// @brief Applies Rectified Linear Unit function to the neurons in the later
/// @param A Neuron layer to which the ReLU is applied.
void ReLU(struct neuron_layer*A){
    int k = A->num_neurons;
    for(int i = 0; i < k; i++){
        if(A->N[i].activation < 0){ A->N[i].activation = 0.0;}
    }
}

/// @brief Does one step of forward propogation
/// @param W Weight matrix
/// @param B Bias matrix
/// @param A1 Initial neuron layer
/// @param A2 Next neuron layer
void forward_propogate_step(struct weight_matrix*W,struct bias_matrix*B, struct neuron_layer* A1, struct neuron_layer*A2){
    matrix_dot_product(A1,W,A2);
    add_matricies(A2,B);
    ReLU(A2);
}

/// @brief Applies softmax to the final output layer.
/// @param A Final output layer
void softmax(struct neuron_layer* A){
    int k = A->num_neurons;
    float expsum = 0.0f;
    for (int i = 0; i < k; i++){
        expsum += exp(A->N[i].activation);
    }
}


float* one_hot_encode(int k){
    float* final = malloc(sizeof(float)*10);
    for(int i = 0; i < 10; i++){
        if(i == k){final[i] = (float)k;}
        else{final[i] = (float)0;}
    }
    return final;
}

float* loss_function(struct neuron_layer* final_layer, int k){
    float* loss = malloc(final_layer->num_neurons*sizeof(float));
    float* j = one_hot_encode(k);
    for(int i = 0;i<final_layer->num_neurons;i++){
        loss[i] = pow(final_layer->N[i].activation-j[i],2);
    }
    free(j);
    return loss;
}

void back_propogate_step(struct neuron_layer* AL,struct bias_matrix*dB,struct weight_matrix* dW, int k){
    // Apply back prop once
    float* dZ = loss_function(AL,k);
    for (int i = 0; i < dW->rows; i++){
        for (int j = 0; j < dW->cols; j++){
            dW->weights[i][j] = dZ[i]*AL->N[j].activation;
            printf("%3f",dW->weights[i][j]);
        }
        printf("\n");
    }
    
}


int main(){
    struct neuron_layer*A1 = layer_constructor(784);
    struct neuron_layer*A2 = layer_constructor(10);
    struct neuron_layer*A3 = layer_constructor(10);
    struct weight_matrix*W1 = weight_matrix_constructor(10,784);
    struct weight_matrix*dW1 = weight_matrix_constructor(10,784);
    struct weight_matrix*W2 = weight_matrix_constructor(10,10);
    struct weight_matrix*dW2 = weight_matrix_constructor(10,10);
    struct bias_matrix*B1 = bias_matrix_constructor(10);
    struct bias_matrix*dB1 = bias_matrix_constructor(10);
    struct bias_matrix*B2 = bias_matrix_constructor(10);
    struct bias_matrix*dB2 = bias_matrix_constructor(10);

    for(int i = 0; i < A1->num_neurons;i++){
        printf("%f\n",A1->N[i].activation);
    }

    printf("\n""\n");
    for(int i = 0; i < A2->num_neurons;i++){
        printf("%f\n",A2->N[i].activation);
    }

    printf("\n""\n");
    for(int i = 0; i < A3->num_neurons;i++){
        printf("%f\n",A3->N[i].activation);
    }

    int j = 0;
    FILE* file = fopen("data/t10k-labels.idx1-ubyte", "r");
    unsigned char* label_array = get_image_labels(file);
    printf("\n\n");
    printf("%d\n",label_array[j]);
    printf("\n\n");
    float* star = one_hot_encode(label_array[j]);
    for(int i = 0; i< 10; i++){
        printf("%f\n",star[i]);
    }

    file = fopen("data/t10k-images.idx3-ubyte", "rb");
    struct pixel_data* activations = get_image_pixel_data(file);
    for (int i = (784*j+1); i <= (784*(j+1)); i++){
        if(i%28 == 0){
            printf("\n");
            if (activations->neuron_activation[i] > 1) {
                printf("# ");  // Brighter pixel
            } else {
                printf(". ");  // Darker pixel
            }
        }else{
            if (activations->neuron_activation[i] > 1) {
                printf("# ");  // Brighter pixel
            } else {
                printf(". ");  // Darker pixel
            }
        }
    }

    for (int i = 0; i <= 784; i++){
        float f = activations->neuron_activation[i];
        A1->N[i].activation = f;
        printf("%f\n",A1->N[i].activation);
    }
    forward_propogate_step(W1,B1,A1,A2);
    forward_propogate_step(W2,B2,A2,A3);
    softmax(A3);

    printf("\n""\n");
    for(int i = 0; i < A3->num_neurons;i++){
        printf("%f\n",A3->N[i].activation);
    }

    back_propogate_step(A3,dB1,dW2,10);
    
    return 1;
}