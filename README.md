# Neural Network Implementation in C

This project implements a neural network in C, handling forward and backward propagation with matrix operations. Additionally, it includes a custom IDX file parser for processing the MNIST dataset.

## Features
- Fully connected neural network implementation
- Matrix operations (addition, multiplication, transposition)
- Activation functions (ReLU, softmax)
- Custom IDX file parser for reading MNIST data
- Memory management for image and label data

## Dependencies
- Standard C libraries: `stdio.h`, `stdlib.h`, `stdint.h`, `math.h`, `string.h`

## IDX File Parsing
A custom implementation of an IDX file parser (`idx-file-parser.h` and `idx-file-parser.c`) is included to load MNIST images and labels.

### References
- `.idx` format reference: Bin-Al Sadiq (GitHub)
- `mnist-neural-network-in-plain-C`: AndrewCarterUK (GitHub)
- IDX file structure documented on [Yann LeCun's website](https://yann.lecun.com/exdb/mnist/)

### IDX File Structure
#### Label File
| Offset | Type            | Value | Description   |
|--------|----------------|-------|---------------|
| 0000   | 32-bit Integer | 2049  | Magic number  |
| 0004   | 32-bit Integer | 60000 | Number of labels |
| 0008   | Unsigned Byte  | ??    | Label         |

#### Image File
| Offset | Type            | Value | Description   |
|--------|----------------|-------|---------------|
| 0000   | 32-bit Integer | 2051  | Magic number  |
| 0004   | 32-bit Integer | 10000 | Number of images |
| 0008   | 32-bit Integer | 28    | Rows         |
| 0012   | 32-bit Integer | 28    | Columns      |
| 0016   | Unsigned Byte  | ??    | Pixel values |

### IDX Parser Functions
- `big_to_little_endian(uint32_t value)`: Converts big-endian values to little-endian.
- `get_image_labels(FILE* file)`: Reads labels from an IDX label file.
- `get_image_pixel_data(FILE* file)`: Reads pixel data from an IDX image file.
- `image_label_finalizer(unsigned char* label_array)`: Frees memory allocated for labels.
- `image_data_finalizer(struct pixel_data* data)`: Frees memory allocated for image data.

## Usage
1. Compile the program using GCC:
   ```sh
   make
   ```
2. Run the executable:
   ```sh
   ./NN-data-structure.exe
   ```

## Trained Neural Network on MNIST
This neural network consists of:
- 784 input neurons
- 32 neurons in the hidden layer
- 10 output neurons

The network was trained using the cross-entropy loss function with an epoch count of 5 and a learning rate of 0.001.

### Training Results
```sh
Total Correct Predictions: 9484
Accuracy of the Model: 9484/10000
```

## Sample Results
```sh
 Total Correct predictions: 9484


 The Accuracy of the model is: 9484/10000

. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . # # # # # # # # . . . . . . . . .
. . . . . . . . . # # # # # # # # # # # # . . . . . . . 
. . . . . . . . . # # # # # # # # # # # # # . . . . . .
. . . . . . . . # # # # # # # . . # # # # # # . . . . .
. . . . . . . . # # # # . . . . . . . # # # # . . . . .
. . . . . . . . # # # . . . . . . . . . # # # . . . . .
. . . . . . . . # # # . . . . . . . . . # # # . . . . .
. . . . . . . # # # # . . . . . . . . . # # # . . . . .
. . . . . . . # # # . . . . . . . . . . # # # . . . . . 
. . . . . . . # # # . . . . . . . . . . # # # . . . . .
. . . . . . . # # . . . . . . . . . . . # # # . . . . .
. . . . . . # # # . . . . . . . . . . . # # # . . . . .
. . . . . . # # # . . . . . . . . . . . # # # . . . . .
. . . . . . # # # . . . . . . . . . . # # # # . . . . .
. . . . . . # # # . . . . . . . . . # # # # # . . . . .
. . . . . . # # # . . . . . . . . # # # # # . . . . . . 
. . . . . . . # # # . . . . . # # # # # # . . . . . . .
. . . . . . . # # # # # # # # # # # # # # . . . . . . .
. . . . . . . # # # # # # # # # # # # . . . . . . . . .
. . . . . . . . # # # # # # # # # . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
Prediction:0
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . # # # . . . . . . . . . . . . . 
. . . . . . . . . . . # # # # . . . . . . . . . . . . .
. . . . . . . . . . # # # # # . . . . . . . . . . . . .
. . . . . . . . . . # # # . . . . . . . . . . . . . . .
. . . . . . . . . . # # . . . . . . . . . . . . . . . . 
. . . . . . . . . # # # . . . . . . . . . . . . . . . .
. . . . . . . . . # # # . . . . . . . . . . . . . . . .
. . . . . . . . . # # . . . . . . . . . . . . . . . . . 
. . . . . . . . . # # . . . . . . # # # # # . . . . . .
. . . . . . . . # # # . . . . # # # # # # # # . . . . .
. . . . . . . . # # # . . . # # # # # . . # # # . . . .
. . . . . . . . # # # . . . # # # . . . . # # # . . . .
. . . . . . . . # # # . . # # # . . . . . . # # . . . . 
. . . . . . . . # # # . . # # # . . . . . . # # . . . .
. . . . . . . . # # # . . # # # . . . . . # # # . . . .
. . . . . . . . . # # # . # # # . . . . . # # # . . . .
. . . . . . . . . # # # . . # # # # # # # # # . . . . .
. . . . . . . . . . # # # # # # # # # # # # . . . . . .
. . . . . . . . . . # # # # # # # # # # # . . . . . . .
. . . . . . . . . . . . # # # # # # # # . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
Prediction:6
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . # # # # # # . . . . . . . . .
. . . . . . . . . . . . . # # # # # # # . . . . . . . .
. . . . . . . . . . . . # # # # # # # # . . . . . . . .
. . . . . . . . . . . . # # # # # # # # . . . . . . . .
. . . . . . . . . . . # # # . # # # # # . . . . . . . . 
. . . . . . . . . . . # # # # # # # # # . . . . . . . .
. . . . . . . . . . . # # # # # # # # # . . . . . . . .
. . . . . . . . . . . # # # # # # # # # . . . . . . . .
. . . . . . . . . . . . # # # # # # # # . . . . . . . .
. . . . . . . . . . . . . . # # # # # . . . . . . . . .
. . . . . . . . . . . . . . # # # # # . . . . . . . . . 
. . . . . . . . . . . . . # # # # # . . . . . . . . . .
. . . . . . . . . . . . # # # # # . . . . . . . . . . .
. . . . . . . . . . . . # # # # . . . . . . . . . . . .
. . . . . . . . . . . # # # # # . . . . . . . . . . . .
. . . . . . . . . # # # # # . . . . . . . . . . . . . .
. . . . . . . . . # # # # . . . . . . . . . . . . . . . 
. . . . . . . . # # # # # . . . . . . . . . . . . . . .
. . . . . . . . # # # . . . . . . . . . . . . . . . . . 
. . . . . . . . # # # . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
Prediction:9

```

This model is 94.84% accurate.