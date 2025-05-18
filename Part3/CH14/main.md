# Sparse matrix computation

## q1

### COO

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|:----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| rowIdx | 0 | 0 | 1 | 2 | 2 | 3 | 3 |
| colIdx | 0 | 2 | 2 | 1 | 2 | 0 | 3 |
| value | 1 | 7 | 8 | 4 | 3 | 2 | 1 |

### CSR

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|:----:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| rowPtrs | 0 | 2 | 3 | 5 | 7 |
| colIdx | 0 | 2 | 2 | 1 | 2 | 0 | 3 |
| value | 1 | 7 | 8 | 4 | 3 | 2 | 1 |

### ELL

| idx    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|:---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| colIdx | 0 | 2 | 1 | 0 | 2 | * | 2 | 3 |
| value  | 1 | 8 | 4 | 2 | 7 | * | 3 | 1 |

### JDS

| idx      | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|:--------:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| iterPtr  | 0 | 4 | 7 |   |   |   |   |
| colIdx   | 0 | 1 | 0 | 2 | 2 | 2 | 3 |
| value    | 1 | 4 | 2 | 8 | 7 | 3 | 1 |

## q2

### COO

Z * 3 integers

### CSR

(m + 1) + 2 * Z

### ELL

missing the max number of a nonzeros row

set it as max of B elements for a row, it should be B * m * 2

### JDS

missing the max number of a nonzeros row

set it as max of B elements for a row, it should be B + 2 * Z + m

## q3 - q5

check Part3\CH14\code\sprase_matrix_computation.cu