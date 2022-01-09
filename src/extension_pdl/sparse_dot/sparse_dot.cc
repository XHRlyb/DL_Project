#include <paddle/extension.h>
#include <utility>
#include <list>

/* CUDA Declaration */

std::vector<paddle::Tensor> csr_dot_csc_cuda(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2_indices,
    paddle::Tensor t2_indptr,
    paddle::Tensor t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
);


std::vector<paddle::Tensor> csr_dot_diag_cuda(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
);


#define CHECK_CUDA(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a CUDA tensor")
#define CHECK_CPU(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU tensor")
// #define CHECK_CONTIGUOUS(x) PD_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CONTIGUOUS(x) PD_CHECK(x.is_initialized(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/* CSR dot CSC Implementation */

std::vector<paddle::Tensor> csr_dot_csc_cpu(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2_indices,
    paddle::Tensor t2_indptr,
    paddle::Tensor t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
){
    CHECK_CPU(t1_indices);
    CHECK_CPU(t1_indptr);
    CHECK_CPU(t1_data);
    CHECK_CPU(t2_indices);
    CHECK_CPU(t2_indptr);
    CHECK_CPU(t2_data);

    std::list<int64_t> out_indices_list[batch_size * out_h];
    std::list<float> out_data_list[batch_size * out_h];
    // auto out_indptr = paddle::zeros({}, t1_indptr.type());
    std::vector<int64_t> shape;
    shape.push_back(batch_size * out_h + 1);
    auto out_indptr = paddle::Tensor(t1_indptr.place(), shape);
    auto si = out_indptr.size();
    auto* out_idp = out_indptr.mutable_data<int64_t>(out_indptr.place());
    for (int i = 0; i < si; ++i) {
        out_idp[i] = static_cast<int64_t>(0);
    }

    auto* t1_indptr_acc = t1_indptr.data<int64_t>();
    auto* t2_indptr_acc = t2_indptr.data<int64_t>();
    auto* t1_indices_acc = t1_indices.data<int64_t>();
    auto* t2_indices_acc = t2_indices.data<int64_t>();

    auto* t1_data_acc = t1_data.data<float>();
    auto* t2_data_acc = t2_data.data<float>();

    // auto t1_indptr_acc = t1_indptr.accessor<int64_t, 1>();
    // auto t2_indptr_acc = t2_indptr.accessor<int64_t, 1>();
    // auto t1_indices_acc = t1_indices.accessor<int64_t, 1>();
    // auto t2_indices_acc = t2_indices.accessor<int64_t, 1>();

    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < out_h; i++)
        {
            int64_t t1_start = t1_indptr_acc[b * out_h + i];
            int64_t t1_stop = t1_indptr_acc[b * out_h + i + 1];
            int64_t row_nnz = 0;

            for (int64_t j = 0; j < out_w; j++)
            {
                int64_t t2_start = t2_indptr_acc[b * out_w + j];
                int64_t t2_stop = t2_indptr_acc[b * out_w + j + 1];

                float outp = 0;//paddle::zeros({}, t1_data.type());
                int64_t t1_ptr_idx = t1_start;
                int64_t t2_ptr_idx = t2_start;

                while (t1_ptr_idx < t1_stop && t2_ptr_idx < t2_stop)
                {
                    int64_t t1_cur_indice = t1_indices_acc[t1_ptr_idx];
                    int64_t t2_cur_indice = t2_indices_acc[t2_ptr_idx];
                    if (t1_cur_indice == t2_cur_indice)
                    {
                        auto tmp = t1_data_acc[t1_ptr_idx] * t2_data_acc[t2_ptr_idx];
                        // auto tmp_acc = tmp.accessor<float, 1>();
                        // outp += tmp_acc[0];
                        outp += tmp;
                        t1_ptr_idx++;
                        t2_ptr_idx++;
                    }
                    else if (t1_cur_indice < t2_cur_indice)
                        t1_ptr_idx++;
                    else
                        t2_ptr_idx++;
                }
                if (outp != 0)
                {
                    out_data_list[b * out_h + i].push_back(outp);
                    out_indices_list[b * out_h + i].push_back(j);
                    row_nnz++;
                }
            }
            out_idp[b * out_h + i + 1] = out_idp[b * out_h + i] + row_nnz;
        }
    }

    auto* out_indptr_acc = out_indptr.data<int64_t>();
    int64_t nnz = out_indptr_acc[-1];

    // auto out_indices = paddle::zeros({nnz}, t1_indices.type());
    std::vector<int64_t> shp;
    shp.push_back(nnz);
    auto out_indices = paddle::Tensor(t1_indices.place(), shp);
    si = out_indices.size();
    auto* out_idc = out_indices.mutable_data<int64_t>(out_indices.place());
    for (int i = 0; i < si; ++i) {
        out_idc[i] = static_cast<int64_t>(0);
    }

    // auto out_data = paddle::zeros({nnz}, t1_data.type());
    auto out_data = paddle::Tensor(t1_data.place(), shp);
    si = out_data.size();
    auto* out_d = out_data.mutable_data<int64_t>(out_data.place());
    for (int i = 0; i < si; ++i) {
        out_d[i] = static_cast<int64_t>(0);
    }

    int64_t idx = 0;
    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < out_h; i++)
        {
            auto * tmp_indices_list = &out_indices_list[b * out_h + i];
            auto * tmp_data_list = &out_data_list[b * out_h + i];
            while (!tmp_indices_list->empty() && !tmp_data_list->empty())
            {
                out_idc[idx] = tmp_indices_list->front();
                tmp_indices_list->pop_front();
                out_d[idx] = tmp_data_list->front();
                tmp_data_list->pop_front();
                idx++;
            }
        }
    }

    return {out_indices, out_indptr, out_data};
}


std::vector<paddle::Tensor> csr_dot_csc_dense_cuda_wrapper(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2_indices,
    paddle::Tensor t2_indptr,
    paddle::Tensor t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
){
    CHECK_INPUT(t1_indices);
    CHECK_INPUT(t1_indptr);
    CHECK_INPUT(t1_data);
    CHECK_INPUT(t2_indices);
    CHECK_INPUT(t2_indptr);
    CHECK_INPUT(t2_data);
    return csr_dot_csc_cuda(t1_indices, t1_indptr, t1_data,
                            t2_indices, t2_indptr, t2_data,
                            batch_size, out_h, out_w);
}


std::vector<paddle::Tensor> csr_dot_csc(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2_indices,
    paddle::Tensor t2_indptr,
    paddle::Tensor t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    if (t1_indices.place()==paddle::PlaceType::kGPU)
        throw std::runtime_error("Unexpected cuda tensor in sparse dot sparse -> sparse computation.");
    else
        return csr_dot_csc_cpu(t1_indices, t1_indptr, t1_data, t2_indices, t2_indptr, t2_data, batch_size, out_h, out_w);
}

std::vector<paddle::Tensor> csr_dot_csc_dense_cuda(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2_indices,
    paddle::Tensor t2_indptr,
    paddle::Tensor t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    return csr_dot_csc_dense_cuda_wrapper(t1_indices, t1_indptr, t1_data, t2_indices, t2_indptr, t2_data,
                                          batch_size, out_h, out_w);
}

/* CSR dot diag implementation */

paddle::Tensor clone(paddle::Tensor t) {
    auto type=t.type();
    if(type==paddle::DataType::FLOAT32) {
        return t.copy_to<float>(t.place());
    } else if(type==paddle::DataType::INT64) {
        return t.copy_to<int64_t>(t.place());
    } else {
        throw 1;
    }
}

std::vector<paddle::Tensor> csr_dot_diag_cpu(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    CHECK_CPU(t1_indices);
    CHECK_CPU(t1_indptr);
    CHECK_CPU(t1_data);
    CHECK_CPU(t2);
    auto outp_indices = clone(t1_indices);
    auto outp_indptr = clone(t1_indptr);
    // auto out_indptr = t1_indptr.copy_to(t1_indptr.place());
    // auto outp_indices = clone<type>(t1_indices);
    // auto outp_indptr = paddle::clone(t1_indptr);
    auto shp = t2.shape();
    auto shp1 = shp[0], shp2 = shp[1];
    // auto outp_data = paddle::zeros_like(t1_data);
    auto outp_data = paddle::Tensor(t1_data.place(), t1_data.shape());
    auto si = outp_data.size();
    auto* outp_data_c = outp_data.mutable_data<float>(outp_data.place());
    for (int i = 0; i < si; ++i) {
        outp_data_c[i] = static_cast<float>(0);
    }

    auto* t1_data_acc = t1_data.data<float>();
    auto* t2_acc = t2.data<float>();


    auto* t1_indptr_acc = t1_indptr.data<int64_t>();
    auto* t1_indices_acc = t1_indices.data<int64_t>();

    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < out_h; i++)
        {
            int64_t start = t1_indptr_acc[b * out_h + i];
            int64_t stop = t1_indptr_acc[b * out_h + i + 1];
            for (int64_t data_idx = start; data_idx < stop; data_idx++)
            {
                int64_t row_idx = t1_indices_acc[data_idx];
                outp_data_c[data_idx] = t1_data_acc[data_idx] * t2_acc[b * shp2 + row_idx];
            }
        }
    }
    return {outp_indices, outp_indptr, outp_data};
}


std::vector<paddle::Tensor> csr_dot_diag_cuda_wrapper(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    CHECK_INPUT(t1_indices);
    CHECK_INPUT(t1_indptr);
    CHECK_INPUT(t1_data);
    CHECK_INPUT(t2);
    return csr_dot_diag_cuda(t1_indices, t1_indptr, t1_data, t2, batch_size, out_h, out_w);
}


std::vector<paddle::Tensor> csr_dot_diag(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    if (t1_indices.place() == paddle::PlaceType::kGPU)
        return csr_dot_diag_cuda_wrapper(t1_indices, t1_indptr, t1_data, t2, batch_size, out_h, out_w);
    else
        return csr_dot_diag_cpu(t1_indices, t1_indptr, t1_data, t2, batch_size, out_h, out_w);

}

/* PyBind Interface 
PYBIND11_MODULE(sparse_dot, m) {
  m.def("csr_dot_csc", &csr_dot_csc, "csr sparse matrix dot csc sparse matrix");
  m.def("csr_dot_csc_dense_cuda", &csr_dot_csc_dense_cuda,
        "cuda implementation of csr sparse matrix dot csc sparse matrix, result is dense");
  m.def("csr_dot_diag", &csr_dot_diag, "csr sparse matrix dot a diagonal of dense vector");
}
*/

// std::vector<std::vector<int64_t>> CsrDotCscInferShape(std::vector<int64_t> x_shape) {
//   return {x_shape};
// }
// std::vector<paddle::DataType> CsrDotCscInferDtype(paddle::DataType x_dtype) {
//   return {x_dtype};
// }

// PD_BUILD_OP(csr_dot_csc)
//     .Inputs({"X"})
//     .Outputs({paddle::Vec("Out")})
//     .Attrs({"t1_indices: paddle::Tensor",
//     "t1_indptr: paddle::Tensor",
//     "t1_data: paddle::Tensor",
//     "t2_indices: paddle::Tensor",
//     "t2_indptr: paddle::Tensor",
//     "t2_data: paddle::Tensor",
//     "batch_size: int64_t",
//     "out_h: int64_t",
//     "out_w: int64_t"})
//     .SetKernelFn(PD_KERNEL(csr_dot_csc))
//     .SetInferShapeFn(PD_INFER_SHAPE(CsrDotCscInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(CsrDotCscInferDtype));

// PD_BUILD_OP(csr_dot_diag)
//     .Inputs({"X"})
//     .Outputs({paddle::Vec("Out")})
//     .Attrs({"t1_indices: paddle::Tensor",
//     "t1_indptr: paddle::Tensor",
//     "t1_data: paddle::Tensor",
//     "t2: paddle::Tensor",
//     "batch_size: int64_t",
//     "out_h: int64_t",
//     "out_w: int64_t"})
//     .SetKernelFn(PD_KERNEL(csr_dot_diag))
//     .SetInferShapeFn(PD_INFER_SHAPE(CsrDotCscInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(CsrDotCscInferDtype));

// PD_BUILD_OP(csr_dot_csc_dense_cuda)
//     .Inputs({"X"})
//     .Outputs({paddle::Vec("Out")})
//     .Attrs({"t1_indices: paddle::Tensor",
//     "t1_indptr: paddle::Tensor",
//     "t1_data: paddle::Tensor",
//     "t2_indices: paddle::Tensor",
//     "t2_indptr: paddle::Tensor",
//     "t2_data: paddle::Tensor",
//     "batch_size: int64_t",
//     "out_h: int64_t",
//     "out_w: int64_t"})
//     .SetKernelFn(PD_KERNEL(csr_dot_csc_dense_cuda))
//     .SetInferShapeFn(PD_INFER_SHAPE(CsrDotCscInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(CsrDotCscInferDtype));

