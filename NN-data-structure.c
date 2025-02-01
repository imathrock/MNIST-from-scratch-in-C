#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>

#include"idx-file-parser.h"

#define BATCH_SIZE 32

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
        for (unsigned int i = 0; i < sizeof(Layer->Weights)/sizeof(float*); i++) {
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
    printf("forward propogating\n");
    for(int i = 0; i < L->rows; i++){
        A2->activations[i] = 0.0;
        for(int j = 0; j < L->cols; j++){
            A2->activations[i] += L->Weights[i][j]*A1->activations[j];
        }
        A2->activations[i] += L->biases[i];
    }
    printf("done\n");
}

/// @brief Applies ReLU to the activations
/// @param A 
void ReLU(struct activations*A){
    for (int i = 0; i < A->size; i++)
    {if(A->activations[i] < 0.0){A->activations[i] = 0.0;}}
}

/// @brief Takes Derivative of ReLU and puts it other activation struct
/// @param A 
/// @param Z_ReLU
void ReLU_derivative(struct activations*A){
    for (int i = 0; i < A->size; i++) 
    {A->activations[i] = (A->activations[i] <= 0.0) ? 0.0 : 1.0;}
}

/// @brief Applies Softmax to the activation layer
/// @param A 
void softmax(struct activations* A) {
    printf("Applying softmax\n");
    int k = A->size;
    float max_activation = A->activations[0];
    for (int i = 1; i < k; i++) {
        if (A->activations[i] > max_activation) {
            max_activation = A->activations[i];
        }
    }
    float expsum = 0.0f;
    for (int i = 0; i < k; i++) {
        A->activations[i] = exp(A->activations[i] - max_activation); // Numerical stability
        expsum += A->activations[i];
    }
    if (expsum == 0.0f) {
        perror("Softmax error: expsum is zero");
        exit(1);
    }
    for (int i = 0; i < k; i++) {
        A->activations[i] /= expsum;
    }
    printf("Applied\n");
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
void loss_function(struct activations* dZ_loss,struct activations* Fl, int k) {
    float* j = one_hot_encode(k);
    if (j == NULL) {
        printf("Failed to allocate memory for one-hot encoding\n");
        exit(1);
    }
    for (int i = 0; i < Fl->size; i++) {
        dZ_loss->activations[i] = Fl->activations[i] - j[i];
    }
    free(j);
}

/// @brief Calcualtes Gradient in activations given previous gradient.
/// @param dZ_curr The gradient to be calculated
/// @param L Weights and biases of the layer in front
/// @param dZ_prev loss function of layer in front
/// @param A_curr ReLU derivative of the current layer
void calc_grad_activation(struct activations* dZ_curr,struct layer*L,struct activations* dZ_prev,struct activations*A_curr){
    if(dZ_curr->size != A_curr->size){perror("The ReLU deriv and n-1 grad activation matricies do not match");exit(1);}
    if(L->rows != dZ_prev->size){perror("The Layer matricies and gradient layer matricies do not match");exit(1);}
    if(L->cols != dZ_curr->size){perror("The Layer matricies and curr_grad layer matricies do not match");exit(1);}
    printf("calculating gradient activation\n");
    for (int i = 0; i < L->cols; i++){
        dZ_curr->activations[i] = 0.0;
        for (int j = 0; j < L->rows; j++){
            dZ_curr->activations[i] += L->Weights[j][i]*dZ_prev->activations[j];
        }
        dZ_curr->activations[i] *= A_curr->activations[i];
    }
    printf("calculated\n");
}


/// @brief Conducts 1 step of back propogation and also updates parameters immediately
/// @param L Layer's weights and biases
/// @param dL Gradient layer
/// @param dZ Loss function or activation gradient
/// @param A n-1th layer
void back_propogate_step(struct layer*L,struct layer*dL,struct activations* dZ,struct activations* A){
    if(dL->rows != L->rows || dL->cols != L->cols){perror("The Gradient and Layer matrices do not match");exit(1);}
    if(dZ->size != dL->rows){perror("Gradient activation and gradient layer matricies do not match");exit(1);}
    if(A->size != dL->cols){perror("activation and GradientLayer matrices do not match");exit(1);}
    printf("Backpropogating\n");
    float dZsum = 0.0;
    for (int i = 0; i < dZ->size; i++){dZsum += dZ->activations[i];}
    float m = 1/(float)A->size;
    for (int i = 0; i < dL->rows; i++){
        dL->biases[i] = dZsum*m;
        for (int j = 0; j < dL->cols; j++){
            dL->Weights[i][j] = m*dZ->activations[i]*A->activations[j];}
    }
    printf("backpropogated\n");
}

/// @brief Given original weights, biases and gradient, updates all the values accordingly
/// @param L Layer
/// @param dL Gradient
void param_update(struct layer*L,struct layer*dL, float Learning_Rate){
    if(dL->rows != L->rows || dL->cols != L->cols){perror("The Gradient and Layer matrices do not match");exit(1);}
    printf("params_updating\n");
    for (int i = 0; i < dL->rows; i++){
        L->biases[i] += Learning_Rate*dL->biases[i];
        for (int j = 0; j < dL->cols; j++){
            L->Weights[i][j] += Learning_Rate*dL->Weights[i][j];}
    }
    printf("params_updated\n");
}

/// @brief Prints out activation values for debugging
/// @param A 
void print_activations(struct activations*A){
    for(int i = 0; i < A->size; i++){printf("%f\n",A->activations[i]);}    
}

/// @brief Inputs image data into activation struct
/// @param pixel_data 
/// @param k index of image
/// @param A 
void input_data(struct pixel_data* pixel_data,int k,struct activations*A){
    int numpx = pixel_data->rows*pixel_data->cols;
    if (A->size != numpx){perror("Wrong layer passed to input");exit(1);}
    for (int i = 0; i < numpx; i++){
        A->activations[i] = pixel_data->neuron_activation[k][i];
    }
}

void show_image(struct activations*A){
    for (int i = 0; i < 784; i++){
        if(i%28 == 0){
            printf("\n");
            if (A->activations[i] > 1) {
                printf("# ");  // Brighter pixel
            } else {
                printf(". ");  // Darker pixel
            }
        }else{
            if (A->activations[i] > 1) {
                printf("# ");  // Brighter pixel
            } else {
                printf(". ");  // Darker pixel
            }
        }
    }
    printf("\n");
}

int main(){

    // Parser Testing
    FILE* file = fopen("data/train-labels.idx1-ubyte", "r");
    unsigned char* label_array = get_image_labels(file);
    printf("\n");
    printf("%d\n",label_array[0]);
    printf("\n");
    float* star = one_hot_encode(label_array[0]);
    for(int i = 0; i< 10; i++){printf("%f\n",star[i]);}
    file = fopen("data/train-images.idx3-ubyte", "rb");
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    for (int i = 0; i < 784; i++){
        if(i%28 == 0){
            printf("\n");
            if (pixel_data->neuron_activation[0][i] > 1) {
                printf("# ");  // Brighter pixel
            } else {
                printf(". ");  // Darker pixel
            }
        }else{
            if (pixel_data->neuron_activation[0][i] > 1) {
                printf("# ");  // Brighter pixel
            } else {
                printf(". ");  // Darker pixel
            }
        }
    }
    printf("\n");

    int LL1 = 784;
    int LL2 = 64;
    int LL3 = 10;

    struct layer*L1 = init_layer(LL2,LL1);
    struct layer*L2 = init_layer(LL3,LL2);

    struct layer*dL1 = init_layer(LL2,LL1);
    struct layer*dL2 = init_layer(LL3,LL2);

    struct activations*A1 = init_activations(LL1);
    struct activations*A2 = init_activations(LL2);
    struct activations*A3 = init_activations(LL3);

    // struct activations*dZ1 = init_activations(LL1);
    struct activations*dZ2 = init_activations(LL2);
    struct activations*loss = init_activations(LL3);

    float Learning_Rate = 0.00001;
    
    for (int i = 0; i < 10; i++)
    {
        for (int k = (100*i); k < (100*i+1); k++){
            printf("Image number: %d\n", k);
            input_data(pixel_data,k,A1);
            // show_image(A1);
            forward_prop_step(A1,L1,A2);
            ReLU(A2);
            printf("A3 Before:\n");
            print_activations(A3);
            forward_prop_step(A2,L2,A3);
            printf("A3 Fp:\n");
            print_activations(A3);
            ReLU(A3);
            softmax(A3);
            printf("Prediction for: %d\n",label_array[k]);
            print_activations(A3);
            loss_function(loss,A3,label_array[k]);
            back_propogate_step(L2,dL2,loss,A2);
            ReLU_derivative(A2);
            calc_grad_activation(dZ2,L2,loss,A2);
            back_propogate_step(L1,dL1,dZ2,A1);
        }
        param_update(L1,dL1,Learning_Rate);
        param_update(L2,dL2,Learning_Rate);
    }
    


    image_label_finalizer(label_array);

    return 1;
}