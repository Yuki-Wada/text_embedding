#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Core>
#include <immintrin.h>

using namespace std;

vector<vector<int>> get_sg_ns_pairs(
    vector<vector<int>> texts,
    unsigned window_size,
    unsigned ns_count,
    vector<float> vocab_ns_prob
){
    vector<int> indices_in;
    vector<int> indices_out;
    vector<int> labels;
    int text_size;
    int curr_window_size;
    int index_in;
    int index_out;

    mt19937 gen = mt19937(rand());
    discrete_distribution<int> negative_sampler(vocab_ns_prob.begin(), vocab_ns_prob.end());

    for (unsigned i = 0; i < texts.size(); ++i) {
        text_size = texts[i].size();
        for (int j = 0; j < texts[i].size(); ++j) {
            index_in = texts[i][j];
            if (index_in == -1) continue;

            curr_window_size = gen() % window_size + 1;
            for (int k = -curr_window_size; k < curr_window_size; ++k) {
                if (k == 0 || j + k < 0 || j + k >= text_size) continue;
                index_out = texts[i][j + k];
                if (index_out == -1) continue;
                indices_in.push_back(index_in);
                indices_out.push_back(index_out);
                labels.push_back(1);
            }
            for (unsigned k = 0; k < ns_count; ++k) {
                index_out = negative_sampler(gen);
                indices_in.push_back(index_in);
                indices_out.push_back(index_out);
                labels.push_back(0);
            }
        }
    }

    return {indices_in, indices_out, labels};
}

void update_w_naive_impl(
    float* w_in,
    float* w_out,
    unsigned vocab_count,
    unsigned hidden_dim,
    vector<int> indices_in,
    vector<int> indices_out,
    vector<int> labels,
    float lr
) {
    unsigned int label_count = labels.size();
    int index_in, index_out, label;
    float output, tmp_w_out;

    for (unsigned i = 0; i < label_count; ++i) {
        index_in = indices_in[i];
        index_out = indices_out[i];

        output = 0.0f;
        for (unsigned j = 0; j < hidden_dim; ++j) {
            output += w_in[index_in * hidden_dim + j] * w_out[index_out * hidden_dim + j];
        }
        output = 1.0f / (1.0f + exp(-output));

        label = labels[i];
        for (unsigned j = 0; j < hidden_dim; ++j) {
            tmp_w_out = w_out[index_out * hidden_dim + j];
            w_out[index_out * hidden_dim + j] += (label - output) * w_in[index_in * hidden_dim + j] * lr;
            w_in[index_in * hidden_dim + j] += (label - output) * tmp_w_out * lr;
        }
    }
}

void update_w_eigen_impl(
    float* w_in,
    float* w_out,
    unsigned vocab_count,
    unsigned hidden_dim,
    vector<int> indices_in,
    vector<int> indices_out,
    vector<int> labels,
    float lr
) {
    unsigned int label_count = labels.size();
    int index_in, index_out, label;
    float output, tmp_w_out;

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> w_in_eigen(
        w_in, vocab_count, hidden_dim);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> w_out_eigen(
        w_out, vocab_count, hidden_dim);
    for (unsigned i = 0; i < label_count; ++i) {
        index_in = indices_in[i];
        index_out = indices_out[i];

        output = w_in_eigen.row(index_in).dot(w_out_eigen.row(index_out));
        output = 1.0f / (1.0f + exp(-output));

        label = labels[i];
        for (unsigned j = 0; j < hidden_dim; ++j) {
            tmp_w_out = w_out_eigen(index_out, j);
            w_out_eigen(index_out, j) += (label - output) * w_in_eigen(index_in, j) * lr;
            w_in_eigen(index_in, j) += (label - output) * tmp_w_out * lr;
        }
    }
}

void update_w_avx_impl(
    float* w_in,
    float* w_out,
    unsigned vocab_count,
    unsigned hidden_dim,
    vector<int> indices_in,
    vector<int> indices_out,
    vector<int> labels,
    float lr
) {
    const unsigned alignment = 8;

    unsigned int label_count = labels.size();
    int index_in, index_out;
    float diff, output, tmp_w_out;
    unsigned i, j;

    float* lrs = (float *)_mm_malloc(alignment * sizeof(float), 32);
    float* diffs = (float *)_mm_malloc(alignment * sizeof(float), 32);
    float* bufs = (float *)_mm_malloc(alignment * sizeof(float), 32);
    for (i = 0; i < alignment; ++i) lrs[i] = lr;
    __m256 tmp_ws_in, tmp_ws_out;

    for (i = 0; i < label_count; ++i) {
        index_in = indices_in[i];
        index_out = indices_out[i];

        output = 0.0f;
        for (j = 0; j + alignment < hidden_dim; j += alignment) {
            auto z = _mm256_mul_ps(
                _mm256_loadu_ps(w_in + index_in * hidden_dim + j), 
                _mm256_loadu_ps(w_out + index_out * hidden_dim + j)
            );
            z = _mm256_hadd_ps(z, z);
            z = _mm256_hadd_ps(z, z);
            output += ((float*)&z)[0] + ((float*)&z)[4];
        }
        for (; j < hidden_dim; ++j) {
            output += w_in[index_in * hidden_dim + j] * w_out[index_out * hidden_dim + j];
        }

        output = 1.0f / (1.0f + exp(-output));

        diff = float(labels[i]) - output;
        for (j = 0; j < alignment; ++j) diffs[j] = diff;

        for (j = 0; j + alignment < hidden_dim; j += alignment) {
            tmp_ws_in = _mm256_loadu_ps(w_in + index_in * hidden_dim + j);
            tmp_ws_out = _mm256_loadu_ps(w_out + index_out * hidden_dim + j);

            _mm256_store_ps(
                bufs,
                _mm256_add_ps(
                    tmp_ws_out,
                    _mm256_mul_ps(
                        _mm256_mul_ps(
                            _mm256_loadu_ps(diffs),
                            tmp_ws_in
                        ),
                        _mm256_loadu_ps(lrs)
                    )
                )
            );
            for (unsigned k = 0; k < alignment; ++k) w_out[index_out * hidden_dim + j + k] = bufs[k];

            _mm256_store_ps(
                bufs,
                _mm256_add_ps(
                    tmp_ws_in,
                    _mm256_mul_ps(
                        _mm256_mul_ps(
                            _mm256_loadu_ps(diffs),
                            tmp_ws_out
                        ),
                        _mm256_loadu_ps(lrs)
                    )
                )
            );
            for (unsigned k = 0; k < alignment; ++k) w_in[index_in * hidden_dim + j + k] = bufs[k];
        }
        for (; j < hidden_dim; ++j) {
            tmp_w_out = w_out[index_out * hidden_dim + j];
            w_out[index_out * hidden_dim + j] += diff * w_in[index_in * hidden_dim + j] * lr;
            w_in[index_in * hidden_dim + j] += diff * tmp_w_out * lr;
        }
    }
}

#include<iostream>
float* np_to_ptr_test(float* w1, float* w2, unsigned N) {
    float* z = (float *)_mm_malloc(N * sizeof(float), 32);

    float a = 0.0f;
    for (unsigned i = 0; i < N; i += 8) {
        auto z = _mm256_mul_ps(
            _mm256_loadu_ps(w1 + i), 
            _mm256_loadu_ps(w2 + i)
        );
        z = _mm256_hadd_ps(z, z);
        z = _mm256_hadd_ps(z, z);
        a += ((float*)&z)[0] + ((float*)&z)[4];
        cout << "-------------------" << endl;
        cout << a << endl;
    }

    return w1;
}

int main() {
    unsigned N = 32;
    float* w1 = (float *)_mm_malloc(N * sizeof(float), 16);
    float* w2 = (float *)_mm_malloc(N * sizeof(float), 16);
    for (unsigned i = 0; i < N; ++i) {
        w1[i] = w2[i] = i;
    }

    // np_to_ptr_test(w1, w2, N);

    return 0;
}