/*
    The following libraries were referenced:
    .idx by Bin-Al sadiq on github
    mnist-neural-network-in-plain-C by AndrewCarterUK on github
*/
/*
    The following code is completely my own work. The data structure of idx files 
    was given on https://yann.lecun.com/exdb/mnist/ website which was found on 
    AndrewCarterUK's github library on a similar code.
*/

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>

#include"idx-file-parser.h"

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
    if(file == NULL){perror("error opening file");printf("File pointer is null");return 1;}
    // Read magic number
    uint32_t magic_number;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    printf("Magic number in big endian: %u\n", magic_number);
    // Read size
    uint32_t size;
    fread(&size, sizeof(uint32_t), 1, file);
    size = big_to_little_endian(size);
    printf("size of array: %u\n", size);
    printf("%s","FINALIZER NAME: image_label_finalizer");
    
    // allocate space for char array of numbers and write the labels in it
    unsigned char* label_array = malloc(size*sizeof(unsigned char));
    for(int i = 0; i< size; i++){
        fread(&label_array[i], sizeof(label_array[i]), 1, file);
        printf("%s\n",label_array[i]);
    }
    fclose(file);
    return label_array;
}

void image_label_finalizer(unsigned char* label_array,uint32_t size){
    for(int i = 0; i < size; i++){free(label_array[i]);}
    free(label_array);
}