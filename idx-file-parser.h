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
#include<string.h>
#include<math.h>
#include<malloc.h>

typedef struct pixel_data{
    uint8_t** neuron_activation;
    uint32_t size;
    uint32_t rows;
    uint32_t cols;
}pixel_data;

uint32_t big_to_little_endian(uint32_t value);

unsigned char* get_image_labels(FILE*file);

struct pixel_data* get_image_pixel_data(FILE*file);

void image_label_finalizer(unsigned char* label_array);

void image_data_finalizer(struct pixel_data* data);