// This code defines the data structures of each element present in a neural network
// It will contain structs that define neurons, layers of neurons, matricies for weights
// Matricies for biases and some linear algebra functions.

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<assert.h>

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

struct neural_network{
    struct weight_matrix*W1;
    struct weight_matrix*W2;
    struct bias_matrix*B1;
    struct bias_matrix*B2;
    struct weight_matrix*dW1;
    struct weight_matrix*dW2;
    struct bias_matrix*dB1;
    struct bias_matrix*dB2;
};

/// @brief constructs a neuron layer with each neuron containing a random number between 0 and 1 
/// @param num_neurons 
struct neuron_layer* layer_constructor(int num_neurons){
    struct neuron_layer*layer = (struct neuron_layer*)malloc(sizeof(struct neuron_layer));
    layer->num_neurons = num_neurons;
    layer->N = (struct neuron*)malloc(num_neurons*sizeof(struct neuron));
    if(layer->N == NULL){printf("Failed to Allocate memory"); exit(1);}
    for(int i = 0; i < num_neurons; i++){
        layer->N[i].activation = ((float)rand() / (float)RAND_MAX) - 0.5;
    }
    return layer;
}

/// @brief Constructs a matrix of weights with randomly assigned values
/// @param rows
/// @param cols
struct weight_matrix* weight_matrix_constructor(int rows, int cols){
    struct weight_matrix*W = (struct weight_matrix*)malloc(sizeof(struct weight_matrix));
    W->rows = rows;W->cols = cols;
    W->weights = (float**)malloc(rows*sizeof(float*));
    if(W->weights == NULL){printf("Failed to Allocate memory"); exit(1);}
    for(int i = 0; i < rows; i++){
        W->weights[i] = (float*)malloc(cols*sizeof(float));
        if(W->weights[i] == NULL){printf("Failed to Allocate memory"); exit(1);}}
    for(int i = 0; i< rows; i++){
        for(int j = 0; j < cols; j++){W->weights[i][j] = ((float)rand() / RAND_MAX) - 0.5;}}
    return W;
}

/// @brief Constructs a bias matrix with random values
/// @param cols
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
    if (layer->N != NULL) 
    {free(layer->N);
    layer->N = NULL;}
    layer->num_neurons = 0;
}

/// @brief To be used for add biases from the layer's activation values
/// @param a Neuron Layer
/// @param b Bias Matrix
void add_bias_matrix(struct neuron_layer *a, struct bias_matrix *b){
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
    for(int i = 0; i < k; i++){if(A->N[i].activation < 0){ A->N[i].activation = 0.0;}}
}


/// @brief Does one step of forward propogation
/// @param W Weight matrix
/// @param B Bias matrix
/// @param A1 Initial neuron layer
/// @param A2 Next neuron layer
void forward_propogate_step(struct weight_matrix*W,struct bias_matrix*B, struct neuron_layer* A1, struct neuron_layer*A2){
    matrix_dot_product(A1,W,A2);
    add_bias_matrix(A2,B);
    ReLU(A2);
}

/// @brief Applies softmax to the final output layer.
/// @param A Final output layer
void softmax(struct neuron_layer* A){
    int k = A->num_neurons;
    float max_activation = A->N[0].activation;
    for (int i = 1; i < k; i++) 
    {if (A->N[i].activation > max_activation) {max_activation = A->N[i].activation;}}
    float expsum = 0.0f;
    for (int i = 0; i < k; i++) 
    {A->N[i].activation = exp(A->N[i].activation - max_activation);
    expsum += A->N[i].activation;}
    for (int i = 0; i < k; i++) {A->N[i].activation /= expsum;}
}

/// @brief One hot encodes the error function
/// @param k 
/// @return float array with null except kth element as 1
float* one_hot_encode(int k){
    float* final = malloc(sizeof(float)*10);
    for(int i = 0; i < 10; i++){
        if(i == k){final[i] = (float)1;}
        else{final[i] = (float)0;}
    }
    return final;
}

/// @brief Loss function that tells us the error values
/// @param final_layer 
/// @param k size
/// @return float array with loss values
float* loss_function(struct neuron_layer* final_layer, int k) {
    float* loss = malloc(final_layer->num_neurons * sizeof(float));
    if (loss == NULL) {
        printf("Failed to allocate memory for loss array\n");
        exit(1);
    }
    float* j = one_hot_encode(k);
    if (j == NULL) {
        printf("Failed to allocate memory for one-hot encoding\n");
        free(loss);
        exit(1);
    }
    for (int i = 0; i < final_layer->num_neurons; i++) {
        loss[i] = pow(final_layer->N[i].activation - j[i],2);
    }
    free(j);
    return loss;
}


/// @brief conducts back prop on the whole neural network
/// @param NN Weights, biases and gradients
/// @param A3 Final layer
/// @param A2 Hidden layer
/// @param A1 Input layer
void backpropogate(struct neural_network* NN, struct neuron_layer* A3, struct neuron_layer* A2, struct neuron_layer* A1, int label) {
    float* dZ2 = loss_function(A3, label);
    for (int i = 0; i < NN->dW2->rows; i++) {
        printf("dZ2[%d] = %f\n", i, dZ2[i]);
    }
    float dampener_coef = 1.0 / NN->dW2->cols;
    // Update dB2 and dW2
    for (int i = 0; i < NN->dW2->rows; i++) {
        if (i >= NN->dB2->size) {
            printf("Index out of bounds: i = %d, size = %d\n", i, NN->dB2->size);
            free(dZ2);
            exit(1);
        }
        NN->dB2->bias[i] = dampener_coef * dZ2[i];
        for (int j = 0; j < NN->dW2->cols; j++) {
            NN->dW2->weights[i][j] = dampener_coef * (dZ2[i] * A2->N[j].activation);
        }
    }

    // Calculate dZ1
    float* dZ1 = (float*)malloc(sizeof(float) * A2->num_neurons);
    if (dZ1 == NULL) {
        printf("Failed to allocate memory for dZ1\n");
        free(dZ2);
        exit(1);
    }

    for (int i = 0; i < A2->num_neurons; i++) {
        dZ1[i] = 0.0;
        for (int k = 0; k < NN->W2->rows; k++) {
            dZ1[i] += dZ2[k] * NN->W2->weights[k][i];
        }
        dZ1[i] *= (A2->N[i].activation > 0) ? 1.0f : 0.0f;
    }

    // Update dB1 and dW1
    for (int i = 0; i < NN->dW1->rows; i++) {
        NN->dB1->bias[i] = dampener_coef * dZ1[i];
        for (int j = 0; j < NN->dW1->cols; j++) {
            NN->dW1->weights[i][j] = dampener_coef * (dZ1[i] * A1->N[j].activation);
        }
    }

    // Free allocated memory
    free(dZ1);
    free(dZ2);
    printf("backpropogated\n");
    return;
}

/// @brief Updates the weights and biases after back propogation
/// @param NN , whole neural network
/// @param a Learning rate
void update_params(struct neural_network*NN,float a){
    printf("Params updating\n");
    for (int i = 0; i < NN->W1->rows; i++){
        NN->B1->bias[i] += a*NN->dB1->bias[i];
        for (int k = 0; k < NN->W1->cols; k++)
        {NN->W1->weights[i][k] += a*NN->dW1->weights[i][k];}}

    for (int i = 0; i < NN->W2->rows; i++){
        NN->B2->bias[i] += a*NN->dB2->bias[i];
        for (int j = 0; j < NN->W2->cols; j++)
        {NN->W2->weights[i][j] += a*NN->dW2->weights[i][j];}}
    printf("Params updated\n");
}

void train_network(struct neural_network*NN,struct neuron_layer* A3,
struct neuron_layer* A2,struct neuron_layer* A1,struct pixel_data*activations,
unsigned char* label_array,float Learning_rate){

    int j=0;
        for(unsigned int i = (784*j); i < (784*(j+1)); i++){
        A1->N[i].activation = activations->neuron_activation[i]/255.0;
        // printf("%d\n",i);
        }
    
        forward_propogate_step(NN->W1,NN->B1,A1,A2);
        printf("%s\n","Step 1 forward prop");
        printf("\n");
        printf("%s\n","A2 ReLU");
        for(int i = 0; i < A2->num_neurons;i++){
            printf("%f\n",A2->N[i].activation);
        }

        forward_propogate_step(NN->W2,NN->B2,A2,A3);
        printf("%s\n","Step 2 forward prop");
        printf("\n");
        printf("%s\n","A3 fp");
        for(int i = 0; i < A3->num_neurons;i++){
            printf("%f\n",A3->N[i].activation);
        }

        softmax(A3);
        printf("\n");
        printf("%s\n","A3 softmax");
        for(int i = 0; i < A3->num_neurons;i++){
            printf("%f\n",A3->N[i].activation);
        }

        backpropogate(NN,A3,A2,A1,label_array[j]);
        update_params(NN,Learning_rate);
    
}


int main(){
    int L1 = 784;
    int L2 = 32;
    int L3 = 10;
    struct neuron_layer*A1 = layer_constructor(L1);
    struct neuron_layer*A2 = layer_constructor(L2);
    struct neuron_layer*A3 = layer_constructor(L3);

    struct weight_matrix*W1 = weight_matrix_constructor(L2,L1);
    struct weight_matrix*dW1 = weight_matrix_constructor(L2,L1);
    struct weight_matrix*W2 = weight_matrix_constructor(L3,L2);
    struct weight_matrix*dW2 = weight_matrix_constructor(L3,L2);

    struct bias_matrix*B1 = bias_matrix_constructor(L2);
    struct bias_matrix*dB1 = bias_matrix_constructor(L2);
    struct bias_matrix*B2 = bias_matrix_constructor(L3);
    struct bias_matrix*dB2 = bias_matrix_constructor(L3);

    struct neural_network*NN = (struct neural_network*)malloc(sizeof(struct neural_network));
    NN->B1 = B1;NN->B2 = B2;NN->W1 = W1;NN->W2 = W2;
    NN->dB1 = dB1;NN->dB2 = dB2;NN->dW1 = dW1;NN->dW2 = dW2;

    // Parser Testing
    int j = 0;
    FILE* file = fopen("data/train-labels.idx1-ubyte", "r");
    unsigned char* label_array = get_image_labels(file);
    printf("\n");
    printf("%d\n",label_array[j]);
    printf("\n");
    float* star = one_hot_encode(label_array[j]);
    for(int i = 0; i< 10; i++){printf("%f\n",star[i]);}
    file = fopen("data/train-images.idx3-ubyte", "rb");
    struct pixel_data* activations = get_image_pixel_data(file);
    for (int i = (784*j); i < (784*(j+1)); i++){
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
    printf("\n");
    
    // train_network(NN,A3,A2,A1,activations,label_array);

    int Learning_rate = 0.01;

    j++;
        for(unsigned int i = (784*j); i < (784*(j+1)); i++){
        A1->N[i].activation = activations->neuron_activation[i]/255.0;
        }
    
        forward_propogate_step(NN->W1,NN->B1,A1,A2);
        printf("%s\n","Step 1 forward prop");
        printf("\n");
        printf("%s\n","A2 ReLU");
        for(int i = 0; i < A2->num_neurons;i++){
            printf("%f\n",A2->N[i].activation);
        }

        forward_propogate_step(NN->W2,NN->B2,A2,A3);
        printf("%s\n","Step 2 forward prop");
        printf("\n");
        printf("%s\n","A3 fp");
        for(int i = 0; i < A3->num_neurons;i++){
            printf("%f\n",A3->N[i].activation);
        }

        softmax(A3);
        printf("\n");
        printf("%s\n","A3 softmax");
        for(int i = 0; i < A3->num_neurons;i++){
            printf("%f\n",A3->N[i].activation);
        }

        backpropogate(NN,A3,A2,A1,label_array[j]);
        printf("%s\n","Test phrase");

        update_params(NN,Learning_rate);


    j++;
        for(unsigned int i = (784*j); i < (784*(j+1)); i++){
        A1->N[i].activation = activations->neuron_activation[i]/255.0;
        // printf("%d\n",i);
        }
    
        forward_propogate_step(NN->W1,NN->B1,A1,A2);
        printf("%s\n","Step 1 forward prop");
        printf("\n");
        printf("%s\n","A2 ReLU");
        for(int i = 0; i < A2->num_neurons;i++){
            printf("%f\n",A2->N[i].activation);
        }

        forward_propogate_step(NN->W2,NN->B2,A2,A3);
        printf("%s\n","Step 2 forward prop");
        printf("\n");
        printf("%s\n","A3 fp");
        for(int i = 0; i < A3->num_neurons;i++){
            printf("%f\n",A3->N[i].activation);
        }

        softmax(A3);
        printf("\n");
        printf("%s\n","A3 softmax");
        for(int i = 0; i < A3->num_neurons;i++){
            printf("%f\n",A3->N[i].activation);
        }

        backpropogate(NN,A3,A2,A1,label_array[j]);
        update_params(NN,Learning_rate);

    image_label_finalizer(label_array);
    destroy_layer(A1);
    destroy_layer(A2);
    destroy_layer(A3);
    destroy_weight_matrix(W1);
    destroy_weight_matrix(W2);

    return 1;
}