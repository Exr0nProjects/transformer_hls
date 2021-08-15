#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

typedef float val_t;
typedef size_t dim_t;

void swap_dim_t(dim_t *a, dim_t *b) {
    dim_t c = *a;
    *a = *b;
    *b = c;
    // TODO: is xor swap faster?
    //*a ^= *b ^= *a ^= *b;
}

struct Matrix {
    val_t *_data; 
    dim_t  _rows;
    dim_t  _cols;
    bool rowmajor;
};
void matrix_construct(struct Matrix *m, dim_t rows, dim_t cols, val_t data[]) {
    // m should already be declared and allocated
    // assert(sizeof(data) == rows*cols*sizeof(val_t)); // not actually, bc data has decayed into ptr
    m->_data = data;
    m->_rows = rows;
    m->_cols = cols;
    m->rowmajor = true;
}
val_t get(struct Matrix *m, dim_t row, dim_t col) {
    if (m->rowmajor) return m->_data[row * m->_cols + col];
    else             return m->_data[col * m->_rows + row];
}
void set(struct Matrix *m, dim_t row, dim_t col, val_t val) {
    if (m->rowmajor) m->_data[row * m->_cols + col] = val;
    else             m->_data[col * m->_rows + row] = val;
}
struct Matrix * matrix_transpose(struct Matrix *m) {
    m->rowmajor ^= true;
    swap_dim_t(&m->_rows, &m->_cols);
    return m;
}
struct Matrix * matrix_dot(struct Matrix *lhs, struct Matrix *rhs, struct Matrix *out) {
    assert(lhs->_rows == out->_rows);
    assert(lhs->_cols == rhs->_rows);
    assert(rhs->_cols == out->_cols);
    // ideally lhs->rowmajor && !rhs->rowmajor
    
    // NTFS: HOT CODE! optimize me
    // also, I guess the HLS-ify people will have to make a bunch of copies of this fn, based on all the matrix sizes we have?
    for (dim_t r=0; r<out->_rows; ++r) {
        for (dim_t c=0; c<out->_cols; ++c) {
            val_t sum = 0;
            for (dim_t i=0; i<lhs->_cols; ++i) {
                sum += get(lhs, r, i) * get(rhs, i, c);
            }
            set(out, r, c, sum);
        }
    }
    return out;
}
struct Matrix * matrix_add(struct Matrix *lhs, struct Matrix *rhs, struct Matrix *out) {
    assert(lhs->_rows == rhs->_rows);
    assert(lhs->_cols == rhs->_cols);
    assert(rhs->_rows == out->_rows);
    assert(rhs->_cols == out->_cols);
    // ideally lhs->rowmajor = true && rhs->rowmajor. maybe make an add_transposed fn and swap loop order?
    for (dim_t r=0; r<out->_rows; ++r)
        for (dim_t c=0; c<out->_cols; ++c)
            set(out, r, c, get(lhs, r, c) + get(rhs, r, c));
    return out;
}

// TODO: allocate all matrices

// TODO: the actual functions


void matrix_print(struct Matrix *m) {
    // TODO: delete, since this can't be HLS-ified
    for (int i=0; i<m->_rows; ++i) {
        for (int j=0; j<m->_cols; ++j) {
            printf("%6.2f", get(m, i, j));
        } printf("\n");
    } printf("\n");
}
int main() 
{
    // TODO: read weights from file
    // TODO: read inputs from file

    // test dot product and point-wise addition
    val_t _a[3][2] = {{2, 2}, {0, 3}, {0, 4}};
    val_t _b[2][3] = {{2, 1, 2}, {3, 2, 4}};
    val_t _c[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    val_t _d[3][3] = {{-10, -6, -12}, {-9, -6, -12}, {-12, -8, -16}};
    val_t _e[2][2] = {{0, 0}, {0, 0}};
    struct Matrix a, b, c, d, e;
    matrix_construct(&a, 3, 2, (val_t *)_a);
    matrix_construct(&b, 2, 3, (val_t *)_b);
    matrix_construct(&c, 3, 3, (val_t *)_c);
    matrix_construct(&d, 3, 3, (val_t *)_d);
    matrix_construct(&e, 2, 2, (val_t *)_e);

    matrix_dot(&a, &b, &c);
    matrix_print(matrix_add(&c, &d, &c));                   // should be all zeros
    matrix_transpose(&a);
    matrix_transpose(&b);
    matrix_dot(&b, &a, &c);
    matrix_print(matrix_add(&c, matrix_transpose(&d), &c)); // should be all zeros
}
