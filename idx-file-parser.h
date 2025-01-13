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

uint32_t big_to_little_endian(uint32_t value);

unsigned char* get_image_labels(FILE*file);

void image_label_finalizer(unsigned char* label_array,uint32_t size);
