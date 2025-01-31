/*
Functions needed:
    Matrix multiplication
    Matrix addition
    ReLU
    ReLU Derivative
    Forward Propogate 
    softmax

    Loss Function
    one hot encoder

    Back propogate
    Update params
    Train Network
*/

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>

#include"idx-file-parser.h"

/// @brief Sturct containing pointers to Weights and biases of the layers.
typedef struct layer{
    float**Weights;
    float*biases;
    int rows;
    int cols;
}layer;

/// @brief Simple float array that holds activation values of the layers. 
typedef struct activations{
    float*activations;
    int size;
}activations;

/// @brief Initializes a layer struct with guards in place to prevent memory leak in case malloc fails.
/// @param rows number of rows in both bias and weight matrix.
/// @param cols number of columns in weight matrix.
struct layer*init_layer(int rows, int cols){
    int r = rows; int c = cols;
    struct layer* Layer = (struct layer*)malloc(sizeof(layer));
    if (Layer == NULL) {
        perror("Failed to allocate memory for Layer");
        return NULL;
    }
    Layer->cols = cols; Layer->rows = rows;
    Layer->biases = malloc(r*sizeof(float));
    if (Layer->biases == NULL) {
        perror("Failed to allocate memory for Layer->biases");
        free(Layer);
        return NULL;
    }
    Layer->Weights = (float**)malloc(rows * sizeof(float*));
    if (Layer->Weights == NULL) {
        perror("Failed to allocate memory for Layer->Weights");
        free(Layer->biases);
        free(Layer);
        return NULL;
    }
    for(int i = 0; i < r; i++){
        Layer->biases[i] = (float)rand()/((float)RAND_MAX) - 0.5;
        Layer->Weights[i] = (float*)malloc(c*sizeof(float));
        if (Layer->Weights[i] == NULL) {
            perror("Failed to allocate memory for Layer->Weights[i]");
            for (int j = 0; j < i; j++) {free(Layer->Weights[j]);}
            free(Layer->Weights);
            free(Layer->biases);
            free(Layer);
            return NULL;
        }
        for(int j = 0; j < c; j++){
            Layer->Weights[i][j] = (float)rand()/((float)RAND_MAX) - 0.5;
        }
    }
    return Layer;
}

/// @brief Frees the Layer struct.
/// @param Layer 
void free_layer(struct layer*Layer){
    if (Layer == NULL) {return;}
    if (Layer->biases != NULL) {
        free(Layer->biases);
        Layer->biases = NULL; 
    }
    if (Layer->Weights != NULL) {
        for (int i = 0; i < sizeof(Layer->Weights)/sizeof(float*); i++) {
            if (Layer->Weights[i] != NULL) {
                free(Layer->Weights[i]);
                Layer->Weights[i] = NULL;
            }
        }
        free(Layer->Weights);
        Layer->Weights = NULL;
    }
    free(Layer);
}

/// @brief initializes and creates a float array to store the activations in.
/// @param size of array
struct activations*init_activations(int size){
    struct activations*activation_layer = (struct activations*)malloc(sizeof(struct activations));
    if (activation_layer == NULL) {
        perror("Failed to allocate memory for Layer");
        return NULL;
    }
    activation_layer->size = size;
    activation_layer->activations = malloc(size*sizeof(float));
    for (int i = 0; i < size; i++){
        activation_layer->activations[i] = (float)rand()/((float)RAND_MAX) - 0.5;
    }
    return activation_layer;
}

/// @brief Frees the activation struct
/// @param a 
void free_activations(struct activations*a){
    free(a->activations);
    free(a);
}

/// @brief Efficient Forward prop function, Does both 
/// @param A1 Previous activations
/// @param L Layer with weights and biases
/// @param A2 Next activations
void forward_prop_step(struct activations*A1,struct layer*L,struct activations*A2){
    if(A1->size != L->cols){perror("A1's size and L's weight's cols do not match"); exit(1);}
    if(A2->size != L->rows){perror("A2's size and L's weight's rows do not match"); exit(1);}
    for(int i = 0; i < L->rows; i++){
        A2->activations[i] = 0.0;
        for(int j = 0; j < L->cols; j++){
            A2->activations[i] += L->Weights[i][j]*A1->activations[j];
        }
        A2->activations[i] += L->biases[i];
    }
}

/// @brief Applies ReLU to the activations
/// @param A 
void ReLU(struct activations*A){
    for (int i = 0; i < A->size; i++)
    {if(A->activations[i] < 0.0){A->activations[i] = 0.0;}}
}

/// @brief Takes Derivative of ReLU in the same activation struct
/// @param A 
void ReLU_derivative(struct activations*A){
    for (int i = 0; i < A->size; i++) 
    {A->activations[i] = (A->activations[i] < 0.0) ? 0.0 : 1.0;}
}

/// @brief Applies Softmax to the activation layer
/// @param A 
void softmax(struct activations*A){
    int k = A->size;
    float max_activation = A->activations[0];
    for (int i = 1; i < k; i++) 
    {if (A->activations[i] > max_activation) {max_activation = A->activations[i];}}
    float expsum = 0.0f;
    for (int i = 0; i < k; i++) 
    {A->activations[i] = exp(A->activations[i] - max_activation);
    expsum += A->activations[i];}
    for (int i = 0; i < k; i++) {A->activations[i] /= expsum;}
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
float* loss_function(struct activations* Fl, int k) {
    float* loss = malloc(Fl->size * sizeof(float));
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
    for (int i = 0; i < Fl->size; i++) {
        loss[i] = pow(Fl->activations[i] - j[i],2);
    }
    free(j);
    return loss;
}



int main(){

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
    int Learning_rate = 0.5;

    struct layer*L1 = init_layer(32,32);
    struct activations*A1 = init_activations(32);
    
    

    image_label_finalizer(label_array);

    return 1;
}