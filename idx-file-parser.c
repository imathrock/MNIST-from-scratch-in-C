/*
    The following libraries were referenced:
    .idx by Bin-Al sadiq on github
    mnist-neural-network-in-plain-C by AndrewCarterUK on github

    The following code is completely my own work. The data structure of idx files 
    was given on https://yann.lecun.com/exdb/mnist/ website which was found on 
    AndrewCarterUK's github library on a similar code.
*/

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<malloc.h>

#include"idx-file-parser.h"

#define LABEL_MAGIC_NUMBER 0x00000801
#define IMAGE_MAGIC_NUMBER 0x00000803

/// @brief changes big endian to little endian
/// @param value big endian 32 bit integer
/// @return little endian 32 bit integer
uint32_t big_to_little_endian(uint32_t value){
    return ((value & 0xFF000000) >> 24) |
        ((value & 0x00FF0000) >> 8) |
        ((value & 0x0000FF00) << 8) |
        ((value & 0x000000FF) << 24);
}

/*
    The data structure of label idx files is as follows:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  60000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
*/

/// @brief Takes in a file pointer to an idx label file and returns a character array.
/// @param file a file with Labels
/// @return unsigned character array.
unsigned char* get_image_labels(FILE*file){
    if(file == NULL){perror("error opening file");printf("File pointer is null");return NULL;}
    // Read magic number
    uint32_t magic_number;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    printf("Magic number : %u\n", big_to_little_endian(magic_number));
    // Read size
    uint32_t size;
    fread(&size, sizeof(uint32_t), 1, file);
    size = big_to_little_endian(size);
    printf("size of array: %u\n", size);
    
    // allocate space for char array of numbers and write the labels in it
    unsigned char* label_array = malloc(size*sizeof(unsigned char));
    size_t bytes_read = fread(label_array, sizeof(unsigned char), size, file);
    if (bytes_read != size) {
        printf("%s\n","not enough bytes read");
    }
    fclose(file);
    return label_array;
}

/*
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
*/
struct pixel_data* get_image_pixel_data(FILE* file) {
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }
    struct pixel_data* neuron_activations = malloc(sizeof(struct pixel_data));
    if (neuron_activations == NULL) {
        perror("Failed to allocate memory for neuron_activations");
        fclose(file);
        return NULL;
    }
    // Read and validate magic number
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) {
        perror("Failed to read magic number");
        fclose(file);
        free(neuron_activations);
        return NULL;
    }
    magic_number = big_to_little_endian(magic_number);
    if (magic_number != IMAGE_MAGIC_NUMBER) {
        printf("Error: Invalid magic number for image file. Expected %u, got %u.\n", IMAGE_MAGIC_NUMBER, magic_number);
        fclose(file);
        free(neuron_activations);
        return NULL;
    }
    // Read size, rows, and columns
    uint32_t size, rows, cols;
    if (fread(&size, sizeof(uint32_t), 1, file) != 1 ||
        fread(&rows, sizeof(uint32_t), 1, file) != 1 ||
        fread(&cols, sizeof(uint32_t), 1, file) != 1) {
        perror("Failed to read image metadata");
        fclose(file);
        free(neuron_activations);
        return NULL;
    }
    size = big_to_little_endian(size);
    rows = big_to_little_endian(rows);
    cols = big_to_little_endian(cols);

    neuron_activations->size = size;
    neuron_activations->rows = rows;
    neuron_activations->cols = cols;
    // Allocate memory for the 2D array
    neuron_activations->neuron_activation = (uint8_t**)malloc(size * sizeof(uint8_t*));
    if (neuron_activations->neuron_activation == NULL) {
        perror("Failed to allocate memory for neuron_activation");
        fclose(file);
        free(neuron_activations);
        return NULL;
    }
    for (uint32_t i = 0; i < size; i++) {
        neuron_activations->neuron_activation[i] = (uint8_t*)malloc(rows * cols * sizeof(uint8_t));
        if (neuron_activations->neuron_activation[i] == NULL) {
            perror("Failed to allocate memory for neuron_activation[i]");
            fclose(file);
            for (uint32_t j = 0; j < i; j++) {
                free(neuron_activations->neuron_activation[j]);
            }
            free(neuron_activations->neuron_activation);
            free(neuron_activations);
            return NULL;
        }
    }
    // Read pixel data
    for (uint32_t i = 0; i < size; i++) {
        size_t bytes_read = fread(neuron_activations->neuron_activation[i], sizeof(uint8_t), rows * cols, file);
        if (bytes_read != rows * cols) {
            printf("Error: Not enough bytes read for image %u. Expected %u, got %u.\n", i, rows * cols, bytes_read);
            fclose(file);
            for (uint32_t j = 0; j < size; j++) {
                free(neuron_activations->neuron_activation[j]);
            }
            free(neuron_activations->neuron_activation);
            free(neuron_activations);
            return NULL;
        }
    }
    fclose(file);
    return neuron_activations;
}


/// @brief finalizer to free the label array.
/// @param label_array 
void image_label_finalizer(unsigned char* label_array){
    if (label_array != NULL) {
        free(label_array);
        label_array = NULL;
    }
}

/// @brief Finalizer function to free the memory allocated for a pixel_data struct.
/// @param data Pointer to the pixel_data struct to be finalized.
void image_data_finalizer(struct pixel_data* data) {
    if (data == NULL) {
        return;
    }
    if (data->neuron_activation != NULL) {
        for (uint32_t i = 0; i < data->rows; i++) {
            if (data->neuron_activation[i] != NULL) {free(data->neuron_activation[i]);}
        }
        free(data->neuron_activation);
        data->neuron_activation = NULL;
    }
    data->size = 0;
    data->rows = 0;
    data->cols = 0;
}