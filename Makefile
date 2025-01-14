# tutorial referred on website: https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
# Compiler and Flags
CC = gcc
CFLAGS = -I. -Wall -Wextra -g

# Dependencies
DEPS = idx-file-parser.h
OBJ = idx-file-parser.o NN-data-structure.o

# Pattern rule for object files
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Target for the executable
NN-data-struct: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

# Clean rule to remove generated files
.PHONY: clean
clean:
	rm -f *.o NN-data-struct
