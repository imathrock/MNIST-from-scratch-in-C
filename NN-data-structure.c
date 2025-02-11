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
            float scale = sqrt(2.0 / (float)Layer->cols);
            Layer->Weights[i][j] = ((float)rand() / (float)RAND_MAX) * 2 * scale - scale;
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

/// @brief Takes Derivative of ReLU and puts it other activation struct
/// @param A 
/// @param Z_ReLU
void ReLU_derivative(struct activations*A,struct activations*B){
    for (int i = 0; i < A->size; i++) 
    {B->activations[i] = (A->activations[i] <= 0.0) ? 0.0 : 1.0;}
}

/// @brief Applies Softmax to the activation layer
/// @param A 
void softmax(struct activations* A) {
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

/// @brief Computes the cross-entropy loss between predicted activations and the true label.
/// @param Fl Predicted activations (output of the softmax layer).
/// @param k True label (integer between 0 and 9).
/// @return Cross-entropy loss value.
float compute_loss(struct activations* Fl, int k) {
    if (k < 0 || k >= Fl->size) {
        perror("Invalid label index");
        exit(1);
    }
    float predicted_prob = Fl->activations[k];      
    if (predicted_prob <= 0.0f) {
        predicted_prob = 1e-15; 
    }
    return -log(predicted_prob);
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
    for (int i = 0; i < L->cols; i++){
        dZ_curr->activations[i] = 0.0;
        for (int j = 0; j < L->rows; j++){
            dZ_curr->activations[i] += L->Weights[j][i]*dZ_prev->activations[j];
        }
        dZ_curr->activations[i] *= A_curr->activations[i];
    }
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
    float m = 1;
    for (int i = 0; i < dL->rows; i++){
        dL->biases[i] = dZ->activations[i] * m;
        for (int j = 0; j < dL->cols; j++){
            dL->Weights[i][j] = m*dZ->activations[i]*A->activations[j];}
    }
}

/// @brief Given original weights, biases and gradient, updates all the values accordingly
/// @param L Layer
/// @param dL Gradient
void param_update(struct layer*L,struct layer*dL, float Learning_Rate){
    if(dL->rows != L->rows || dL->cols != L->cols){perror("The Gradient and Layer matrices do not match");exit(1);}
    for (int i = 0; i < dL->rows; i++){
        L->biases[i] += Learning_Rate*dL->biases[i];
        for (int j = 0; j < dL->cols; j++){
            L->Weights[i][j] += Learning_Rate*dL->Weights[i][j];}
    }
}

/// @brief Clears the Given layer
/// @param L Layer
void Zero_Layer(struct layer*L,float num){
    if(num>1){perror("Incorrect value passed\n"); exit(1);}
    for (int i = 0; i < L->rows; i++){
        L->biases[i] = 0;
        for (int j = 0; j < L->cols; j++)
            {L->Weights[i][j] = 0;}
    }
}

/// @brief Inputs image data into activation struct
/// @param pixel_data 
/// @param k index of image
/// @param A 
void input_data(struct pixel_data* pixel_data,int k,struct activations*A){
    int numpx = pixel_data->rows*pixel_data->cols;
    if (A->size != numpx){perror("Wrong layer passed to input");exit(1);}
    for (int i = 0; i < numpx; i++){
        A->activations[i] = pixel_data->neuron_activation[k][i]/255.0;
    }
}

/// @brief Gets the largest activation value and returns it
/// @param A 
/// @return index of highest activation
int get_pred_from_softmax(struct activations *A) {
    int max_index = 0;
    float max_value = A->activations[0];
    for (int i = 1; i < A->size; i++) {
        if (A->activations[i] > max_value) {
            max_value = A->activations[i];
            max_index = i;}
    }
    return max_index;
}


/// @brief Prints out activation values for debugging
/// @param A 
void print_activations(struct activations*A){
    for(int i = 0; i < A->size; i++){printf("%f\n",A->activations[i]);}    
}

/// @brief Prints the contents of a layer struct.
/// @param l Pointer to the layer struct to be printed.
void print_layer(const struct layer* l) {
    if (l == NULL) {
        printf("Layer is NULL.\n");
        return;
    }
    printf("Layer dimensions: rows = %d, cols = %d\n", l->rows, l->cols);
    printf("Weights:\n");
    for (int i = 0; i < l->rows; i++) {
        for (int j = 0; j < l->cols; j++) {
            printf("%8.4f ", l->Weights[i][j]);
        }
        printf("\n");
    }
    printf("Biases:\n");
    for (int i = 0; i < l->rows; i++) {
        printf("%8.4f ", l->biases[i]);
    }
    printf("\n");
}


/// @brief Shows the image at kth index.
/// @param pixel_data 
/// @param k 
void show_image(struct pixel_data*pixel_data,int k){
    for (int i = 0; i < 784; i++){
        if(i%28 == 0){
            printf("\n");
            if (pixel_data->neuron_activation[k][i] > 10) {
                printf("# ");
            } else {
                printf(". ");
            }
        }else{
            if (pixel_data->neuron_activation[k][i] > 10) {
                printf("# ");
            } else {
                printf(". ");
            }
        }
    }
    printf("\n");
}

int main(){

    // Parser Testing
    FILE* file = fopen("data/train-labels-idx1-ubyte", "r");
    unsigned char* label_array = get_image_labels(file);
    printf("\n");
    printf("%d\n",label_array[0]);
    printf("\n");
    float* star = one_hot_encode(label_array[0]);
    for(int i = 0; i< 10; i++){printf("%f\n",star[i]);}
    file = fopen("data/train-images-idx3-ubyte", "rb");
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    show_image(pixel_data,0);
    printf("\n");

    int LL1 = 784;
    int LL2 = 128;
    int LL3 = 10;

    struct layer*L1 = init_layer(LL2,LL1);
    struct layer*L2 = init_layer(LL3,LL2);

    struct layer*dL1 = init_layer(LL2,LL1);
    struct layer*dL2 = init_layer(LL3,LL2);

    struct layer*sdL1 = init_layer(LL2,LL1);
    struct layer*sdL2 = init_layer(LL3,LL2);

    struct activations*A1 = init_activations(LL1);
    struct activations*A2 = init_activations(LL2);
    struct activations*A3 = init_activations(LL3);

    struct activations*dA_RELU = init_activations(LL2);
    struct activations*dZ2 = init_activations(LL2);
    struct activations*loss = init_activations(LL3);

    float Learning_Rate = 0.05;
    int epoch = 5;
    int size = pixel_data->size/BATCH_SIZE;
    while(epoch--){
        for (int i = 0; i < size; i++){
            float total_loss = 0.0;
            for (int k = (BATCH_SIZE*i); k < (BATCH_SIZE*(i+1)); k++){
                input_data(pixel_data,k,A1);
                forward_prop_step(A1,L1,A2);
                ReLU(A2);
                forward_prop_step(A2,L2,A3);
                softmax(A3);
                loss_function(loss,A3,label_array[k]);
                back_propogate_step(L2,dL2,loss,A2);
                ReLU_derivative(A2,dA_RELU);
                calc_grad_activation(dZ2,L2,loss,dA_RELU);
                back_propogate_step(L1,dL1,dZ2,A1);
                param_update(sdL1,dL1,1);
                param_update(sdL2,dL2,1);
                total_loss += compute_loss(A3,label_array[k])/BATCH_SIZE;
            }
            printf("\n\nBatch Loss:%f\n",total_loss);
            param_update(L1,sdL1,-Learning_Rate);
            param_update(L2,sdL2,-Learning_Rate);
            Zero_Layer(sdL1,0);
            Zero_Layer(sdL2,0);
        }    
    }

    image_data_finalizer(pixel_data);
    image_label_finalizer(label_array);

    FILE* test_file = fopen("data/t10k-labels-idx1-ubyte", "r");
    unsigned char* test_lbl_arr = get_image_labels(test_file);
    test_file = fopen("data/t10k-images-idx3-ubyte", "rb");
    struct pixel_data* test_pix_data = get_image_pixel_data(test_file);

    printf("\n\nCalculating accuracy:-\n\n");
    
    int correct_pred = 0;
    for (unsigned int k = 0; k < test_pix_data->size; k++){
        input_data(test_pix_data,k,A1);
        forward_prop_step(A1,L1,A2);
        ReLU(A2);
        forward_prop_step(A2,L2,A3);
        softmax(A3);
        if(test_lbl_arr[k] == get_pred_from_softmax(A3)){correct_pred++;}
        if (k%100 == 0){printf(".");}
    }
    printf("\n\n Total Correct predictions: %d\n",correct_pred);
    printf("\n\n The Accuracy of the model is: %d/%d\n",correct_pred,test_pix_data->size);

    input_data(test_pix_data,10,A1);
    show_image(test_pix_data,10);
    forward_prop_step(A1,L1,A2);
    ReLU(A2);
    forward_prop_step(A2,L2,A3);
    softmax(A3);
    printf("Prediction:%d\n",get_pred_from_softmax(A3));
    printf("Answer:%d",test_lbl_arr[10]);

    input_data(test_pix_data,100,A1);
    show_image(test_pix_data,100);
    forward_prop_step(A1,L1,A2);
    ReLU(A2);
    forward_prop_step(A2,L2,A3);
    softmax(A3);
    printf("Prediction:%d\n",get_pred_from_softmax(A3));
    printf("Answer:%d",test_lbl_arr[100]);
    

    input_data(test_pix_data,1100,A1);
    show_image(test_pix_data,1100);
    forward_prop_step(A1,L1,A2);
    ReLU(A2);
    forward_prop_step(A2,L2,A3);
    softmax(A3);
    printf("Prediction:%d\n",get_pred_from_softmax(A3));
    printf("Answer:%d",test_lbl_arr[1100]);

    image_data_finalizer(test_pix_data);
    image_label_finalizer(test_lbl_arr);
    return 1;
}
