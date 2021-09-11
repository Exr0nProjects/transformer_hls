#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

typedef float val_t;
typedef int dim_t;

const dim_t emb_dim = 2;
const int stack_height = 12;

void swap_dim_t(dim_t *a, dim_t *b) {
    dim_t c = *a;
    *a = *b;
    *b = c;
    // TODO: is xor swap faster?
    //*a ^= *b ^= *a ^= *b;
}

/// MATRIX FUNCTIONS
struct Matrix {
    val_t *_data; 
    dim_t  _rows;
    dim_t  _cols;
    int rowmajor;
};

val_t get(val_t *dat, int rowmajor, dim_t rows, dim_t cols, dim_t row, dim_t col) {
    if (rowmajor) return dat[row * cols + col];
    else return dat[col * rows + row];
}
void set(val_t *dat, int rowmajor, dim_t rows, dim_t cols, dim_t row, dim_t col, val_t val) {
    if (rowmajor) dat[row * cols + col] = val;
    else          dat[col * rows + row] = val;
}

void matrix_transpose(int *row_major, dim_t *rows, dim_t *cols){
  *row_major ^= true;
  swap_dim_t(rows, cols);
}

void matrix_print(val_t *m, int row_major, dim_t rows, dim_t cols) {
    // TODO: delete, since this can't be HLS-ified
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            printf("%10.4f", get(m, row_major, rows, cols, i, j));
        } printf("\n");
    } printf("\n");
}

/*dot product stuff*/

void matrix_dot(val_t *lhs, dim_t l_rows, dim_t l_cols, int l_row_major, val_t *rhs, dim_t r_rows, dim_t r_cols, int r_row_major, val_t *out, dim_t out_rows, dim_t out_cols, int out_row_major) {
    assert(l_rows == out_rows);
    assert(l_cols == r_rows);
    assert(r_cols == out_cols);
    // ideally lhs->rowmajor && !rhs->rowmajor
    
    for (dim_t r=0; r<out_rows; ++r) {
        for (dim_t c=0; c<out_cols; ++c) {
            val_t sum = 0;
            for (dim_t i=0; i<l_cols; ++i) {
                sum += get(lhs, l_row_major, l_rows, l_cols, r, i) * get(rhs, r_row_major, r_rows, r_cols, i, c);
            }
            set(out, out_row_major, out_rows, out_cols, r, c, sum);
        }
    }
}
void matrix_add(val_t *lhs, dim_t l_rows, dim_t l_cols, int l_row_major, val_t *rhs, dim_t r_rows, dim_t r_cols, int r_row_major, val_t *out, dim_t out_rows, dim_t out_cols, int out_row_major) {
    assert(l_rows == r_rows);
    assert(l_cols == r_cols);
    assert(r_rows == out_rows);
    assert(r_cols == out_cols);
    // ideally lhs->rowmajor = true && rhs->rowmajor. maybe make an add_transposed fn and swap loop order?
    for (dim_t r=0; r<out_rows; ++r)
        for (dim_t c=0; c<out_cols; ++c)
            set(out, out_row_major, out_rows, out_cols, r, c, get(lhs, l_row_major, l_rows, l_cols, r, c) + get(rhs, r_row_major, r_rows, r_cols, r, c));
}
void add_biases(val_t *m, dim_t m_rows, dim_t m_cols, int m_row_major, val_t *b, dim_t b_rows, dim_t b_cols, int b_row_major) {
  for(dim_t r = 0; r < m_rows; r++){
    for(dim_t c = 0; c < m_cols; ++c){
      set(m, m_row_major, m_rows, m_cols, r, c, get(m, m_row_major, m_rows, m_cols, r, c) + get(b, b_row_major, b_rows, b_cols, r, 0));
    }
  }
}
void matrix_exp(val_t *m, dim_t m_rows, dim_t m_cols, int m_row_major) {  // in place
    for (dim_t r=0; r<m_rows; ++r) 
        for (dim_t c=0; c<m_cols; ++c) 
            set(m, m_row_major, m_rows, m_cols, r, c, (val_t) expf((float) get(m, m_row_major, m_rows, m_cols, r, c)));   // TODO: exp is for f32, which may not be val_t
}

void matrix_divide(val_t *m, dim_t m_rows, dim_t m_cols, int m_row_major, val_t quotient){
  for (dim_t r=0; r<m_rows; ++r) 
        for (dim_t c=0; c<m_cols; ++c)
                set(m, m_row_major, m_rows, m_cols, r, c, get(m, m_row_major, m_rows, m_cols, r, c)/quotient);
}

void pointwise_relu(val_t *m, dim_t m_rows, dim_t m_cols, int m_row_major) {
    for (dim_t r=0; r<m_rows; ++r) 
        for (dim_t c=0; c<m_cols; ++c)
            if (get(m, m_row_major, m_rows, m_cols, r, c) < 0) 
                set(m, m_row_major, m_rows, m_cols, r, c, 0);
}

void casually_masked_softmax(int seq_len, val_t *m, dim_t m_rows, dim_t m_cols, int m_row_major, val_t *scratch, dim_t scratch_rows, dim_t scratch_cols, int scratch_row_major)
{
    assert(m_rows == seq_len);
    assert(m_cols == seq_len);
    //assert(x->rowmajor == false);   // for maximum cache locality
    assert(scratch_cols == seq_len);
    assert(scratch_rows == 1);
    // exp
    matrix_exp(m, m_rows, m_cols, m_row_major);
    // masked sum
    for (dim_t c=0; c<seq_len; ++c) {
        val_t sum = 0;
        for (dim_t r=0; r<=c; ++r)
            sum += get(m, m_row_major, m_rows, m_cols, r, c);
        set(scratch, scratch_row_major, scratch_rows, scratch_cols, 0, c, sum);
    }
    // div
    for (dim_t r=0; r<m_rows; ++r)
        for (dim_t c=0; c<m_cols; ++c) 
            if (r > c) set(m, m_row_major, m_rows, m_cols, r, c, 0);
            else set(m, m_row_major, m_rows, m_cols, r, c, get(m, m_row_major, m_rows, m_cols, r, c) / get(scratch, scratch_row_major, scratch_rows, scratch_cols, 0, c));
}

/*layer_norm - finish doc later*/
void layer_norm(val_t *dat, dim_t dat_rows, dim_t dat_cols, int dat_row_major, val_t * w, dim_t w_rows, dim_t w_cols, int w_row_major, val_t * b, dim_t b_rows, dim_t b_cols, int b_row_major){
  //add assert statements
  assert(w_cols == 1);
  assert(b_cols == 1);
  assert(w_rows == dat_cols);
  assert(b_rows == dat_cols);
  
  matrix_transpose(&dat_row_major, &dat_rows, &dat_cols);
  //printf("%i, %i, %i \n", dat_row_major, dat_rows, dat_cols);
  //find the mean + std for each row
  for(dim_t r = 0; r < dat_rows; ++r){
    val_t mean = 0;
    for(dim_t c = 0; c < dat_cols; ++c){
      mean += get(dat, dat_row_major, dat_rows, dat_cols, r, c);
    }
    mean /= dat_cols;
    val_t std = 0;
    for(dim_t c = 0; c < dat_cols; ++c){
      std += pow(get(dat, dat_row_major, dat_rows, dat_cols, r, c)-mean, 2);
    }
    std /= dat_cols;
    std += 1e-5;
    std = sqrt(std);
    //do the normalization functions
    for(dim_t c = 0; c < dat_cols; ++c){
      set(dat, dat_row_major, dat_rows, dat_cols, r, c, get(w, w_row_major, w_rows, w_cols, r, 0) * (get(dat, dat_row_major, dat_rows, dat_cols, r, c) - mean)/std + get(b, b_row_major, b_rows, b_cols, r, 0));
    }
  }
  
  matrix_transpose(&dat_row_major, &dat_rows, &dat_cols);
}

void feed_forward_network(val_t *m, int m_row_major, val_t * ln_w, int ln_w_row_major, val_t * ln_b, int ln_b_row_major, val_t *fc_w, int fc_w_row_major, val_t *fc_b, int fc_b_row_major, val_t * proj_w, int proj_w_row_major, val_t * proj_b, int proj_b_row_major, val_t *copy_of_m, val_t *aux_m, int aux_m_row_major){
  //assert functions
  layer_norm(m, emb_dim, emb_dim, m_row_major, ln_w, emb_dim, 1, ln_w_row_major, ln_b, emb_dim, 1, ln_b_row_major);

  matrix_dot(fc_w, 4*emb_dim, emb_dim, fc_w_row_major, m, emb_dim, emb_dim, m_row_major, aux_m, 4*emb_dim, emb_dim, aux_m_row_major);
  add_biases(aux_m, 4*emb_dim, emb_dim, aux_m_row_major, fc_b, 4*emb_dim, 1, fc_b_row_major);

  pointwise_relu(aux_m, 4*emb_dim, emb_dim, aux_m_row_major);

  matrix_dot(proj_w, emb_dim, 4*emb_dim, proj_w_row_major, aux_m, 4*emb_dim, emb_dim, aux_m_row_major, m, emb_dim, emb_dim, m_row_major);
  add_biases(m, emb_dim, emb_dim, m_row_major, proj_b, emb_dim, 1, proj_b_row_major);

  matrix_add(m, emb_dim, emb_dim, m_row_major, copy_of_m, emb_dim, emb_dim, m_row_major, m, emb_dim, emb_dim, m_row_major);
}

void self_attention(val_t *m, int m_row_major, val_t * ln_w, int ln_w_row_major, val_t * ln_b, int ln_b_row_major, val_t * attn_w, int attn_w_row_major, val_t * attn_b, int attn_b_row_major, val_t * proj_w, int proj_w_row_major, val_t *proj_b, int proj_b_row_major, val_t* copy_of_m, int copy_of_m_row_major, val_t *aux_m, int aux_m_row_major, val_t * query, int query_row_major, val_t *key, int key_row_major, val_t *value, int value_row_major, val_t * aux_attn_m, int aux_attn_m_row_major, val_t * scratch, int scratch_row_major, val_t * aux_2, int aux_2_row_major, val_t heads){
  layer_norm(m, emb_dim, emb_dim, m_row_major, ln_w, emb_dim, 1, ln_w_row_major, ln_b, emb_dim, 1, ln_b_row_major);

  val_t adim = emb_dim / heads;
  // attn weights/biases
  matrix_dot(attn_w, 3*emb_dim, emb_dim, attn_w_row_major, m, emb_dim, emb_dim, m_row_major, aux_m, 3*emb_dim, emb_dim, aux_m_row_major);
  add_biases(aux_m, 3*emb_dim, emb_dim, aux_m_row_major, attn_b, emb_dim, 1, attn_b_row_major);
  /*aux_m = pointwise_relu(aux_m);
  m = add_biases(matrix_dot(proj_w, aux_m, m), proj_b);*/

  for(int h = 0; h < heads; h++){
    //split into key/query/val
    for(int i = 0; i < adim; i++){
      for(int j = 0; j < emb_dim; j++){
        set(query, query_row_major, adim, emb_dim, i, j, get(aux_m, aux_m_row_major, 3*emb_dim, emb_dim, adim*h + i, j));
      }
    }
    for(int i = 0; i < adim; i++){
      for(int j = 0; j < emb_dim; j++){
        set(key, key_row_major, adim, emb_dim, i, j, get(aux_m, aux_m_row_major, emb_dim*3, emb_dim, i+emb_dim+adim*h, j));
      }
    }
    for(int i = 0; i < adim; i++){
      for(int j = 0; j < emb_dim; j++){
        set(value, value_row_major, adim, emb_dim, i, j, get(aux_m, aux_m_row_major, emb_dim*3, emb_dim, i+2*emb_dim+adim*h, j));
      }
    }

    //matrix_print(query);
    //matrix_print(key);
    dim_t key_rows = adim, key_cols = emb_dim;

    matrix_transpose(&key_row_major, &key_rows, &key_cols);

    matrix_dot(key, key_rows, key_cols, key_row_major, query, adim, emb_dim, query_row_major, aux_attn_m, emb_dim, emb_dim, aux_attn_m_row_major);

    matrix_transpose(&key_row_major, &key_rows, &key_cols);
    //printf("aux attn m \n");
    //matrix_print(aux_attn_m);
    matrix_divide(aux_attn_m, emb_dim, emb_dim, aux_attn_m_row_major, sqrt(adim));
    //matrix_print(aux_attn_m);
    matrix_transpose(&aux_attn_m_row_major, &key_cols, &key_cols);
    casually_masked_softmax(emb_dim, aux_attn_m, emb_dim, emb_dim, aux_attn_m_row_major, scratch, 1, emb_dim, scratch_row_major);
    //aux_attn_m = matrix_transpose(aux_attn_m);
    //printf("softmax\n");
    //matrix_print(aux_attn_m);
    matrix_dot(value, adim, emb_dim, value_row_major, aux_attn_m, emb_dim, emb_dim, aux_attn_m_row_major, query, adim, emb_dim, query_row_major);
    //printf("query \n");
    //matrix_print(query);
    for(int i = 0; i < adim; i++){
      for(int j = 0; j < emb_dim; j++){
        set(m, m_row_major, emb_dim, emb_dim, i+adim*h, j, get(query, query_row_major, adim, emb_dim, i, j));
      }
    }
  }
  //matrix_print(m);
  matrix_dot(proj_w, emb_dim, emb_dim, proj_w_row_major, m, emb_dim, emb_dim, m_row_major, aux_2, emb_dim, emb_dim, aux_2_row_major);
  add_biases(aux_2, emb_dim, emb_dim, aux_2_row_major, proj_b, emb_dim, 1, proj_b_row_major);

  //matrix_print(aux_2);
  matrix_add(aux_2, emb_dim, emb_dim, aux_2_row_major, copy_of_m, emb_dim, emb_dim, m_row_major, m, emb_dim, emb_dim, m_row_major);
}

int main() 
{
    //dot product test
    /*val_t b[8][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9},{10, 11}, {12, 13}, {14, 15}};
    val_t a[2][2] = {{1, 2}, {5, 6}};
    val_t c[2][8] = {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
    val_t d[2][2] = {{5, 7}, {1, 2}};
    matrix_dot((val_t*)b, 8, 2, true, (val_t*) a, 2, 2, true, (val_t*) c, 8, 2, false);
    matrix_print((val_t*)c, false, 8, 2);
    // validate matrix add
    matrix_add((val_t*)a, 2, 2, 1, (val_t*)d, 2, 2, 1, (val_t*)d, 2, 2, 1);
    matrix_print((val_t*)d, 1, 2, 2);
    //validate add add_biases
    val_t bias[2][1] = {{2}, {6}};
    add_biases((val_t*) a, 2, 2, 1, (val_t*)bias, 2, 1, 1);
    matrix_print((val_t*)a, 1, 2, 2);
    //validate matrix_exp
    matrix_exp((val_t*)a, 2, 2, 1);
    matrix_print((val_t*)a, 1, 2, 2);
    //validate matrix_divide
    matrix_divide((val_t*)d, 2, 2, 1, 3);
    matrix_print((val_t*)d, 1, 2, 2);

    //validate pointwise_relu
    val_t r[2][2] = {{1, 10}, {2, -3}};
    pointwise_relu((val_t*)r, 2, 2, 1);
    matrix_print((val_t*) r, 1, 2, 2);

    //validate softmax
    val_t softmax_scratch[1][2] = {{0, 0}};
    casually_masked_softmax(2, (val_t*) r, 2, 2, 1, (val_t*) softmax_scratch, 1, 2, 1);
    matrix_print((val_t*) r, 1, 2, 2);*/

    //layer_norm validation
    /*val_t a[2][2] = {{1, 2}, {5, 9}};
    matrix_print((val_t*) a, true, 2, 2);
    val_t w[2][1] = {{3}, {4}};
    val_t b[2][1] = {{2}, {6}};
    layer_norm((val_t*) a, 2, 2, true, (val_t*) w, 2, 1, true, (val_t*) b, 2, 1, true);
    matrix_print((val_t*) a, true, 2, 2);*/

    //ffn test
    /*val_t x[2][2] = {{1, 2}, {5, 6}};
    val_t ln_w[2][1] = {{3}, {4}};
    val_t ln_b[2][1] = {{2}, {6}};
    val_t fc_w[8][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9},{10, 11}, {12, 13}, {14, 15}};
    val_t fc_b[8][1] = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}};
    val_t proj_w[2][8] = {{0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14, 15}};
    val_t proj_b[2][1] = {{0}, {1}};
    val_t x2[2][2] = {{1, 2}, {5, 6}};
    val_t aux[8][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
    feed_forward_network((val_t*) x, 1, (val_t*) ln_w, 1, (val_t*) ln_b, 1, (val_t*) fc_w, 1, (val_t*) fc_b, 1, (val_t*) proj_w, 1, (val_t*) proj_b, 1, (val_t*) x2, (val_t*) aux, 1);
    matrix_print((val_t*) x, 1, 2, 2);*/
    
    val_t x_attn[2][2] = {{0.01, .02}, {.05, .06}};
    val_t attn_w[6][2] = {{0, .01}, {.02, .03}, {.04, .05}, {.06, .07}, {.08, .09}, {.10, .11}};
    val_t attn_b[6][1] = {{0}, {.01}, {.02}, {.03}, {.04}, {.05}};
    val_t p_w[2][2] = {{0, 1}, {2, 3}};
    val_t p_b[2][1] = {{0}, {1}};
    val_t aux1[6][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}};
    val_t x_copy[2][2] = {{0.01, .02}, {.05, .06}};

    val_t query[1][2] = {{1, 2}};
    val_t key[1][2] = {{1, 2}};
    val_t value[1][2] = {{1, 2}};

    val_t ln_w[2][1] = {{.3}, {.4}};
    val_t ln_b[2][1] = {{.2}, {.6}};

    val_t aux_attn[2][2] = {{0, 0}, {0, 0}};

    val_t aux_2[2][2] = {{0.01, .02}, {.05, .06}};

    val_t scratch2[1][2] = {{1, 2}};

    self_attention((val_t*)x_attn, 1, (val_t*)ln_w, 1, (val_t*)ln_b, 1, (val_t*)attn_w, 1, (val_t*)attn_b, 1, (val_t*)p_w, 1, (val_t*)p_b, 1, (val_t*)x_copy, 1, (val_t*)aux1, 1, (val_t*)query, 1, (val_t*)key, 1, (val_t*)value, 1, (val_t*)aux_attn, 1, (val_t*)scratch2, 1, (val_t*)aux_2, 1, 2);
    matrix_print((val_t*)x_attn, 1, emb_dim, emb_dim);
}
