package llamacpp

// #cgo CFLAGS: -O3 -DNDEBUG -I ../../../../include -I ../../../../ggml/include
// #cgo LDFLAGS: -L ../../../../src  -L ../../../../ggml/src -l llama -l ggml -l ggml-base -l ggml-cpu -l gomp -l common -l m  -l stdc++
// #include <stdlib.h>
// #include "llama.h"
// #include "ggml.h"
// #include "gguf.h"
// extern bool cgo_scheduler_callback(struct ggml_tensor * t, bool ask, void * user_data);
import "C"

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"unsafe"
)

func simpleHash(data []byte) uint32 {
	var hash uint32
	for _, b := range data {
		hash = (hash * 31) + uint32(b)
	}
	return hash
}

func NewGgmlInitParams(memorySize int) C.struct_ggml_init_params {
	return C.struct_ggml_init_params{
		mem_size:   C.size_t(memorySize),
		mem_buffer: nil,
		no_alloc:   false,
	}
}

type GgmlContext struct {
	p *C.struct_ggml_context
}

func NewGgmlContext(params C.struct_ggml_init_params) (*GgmlContext, error) {
	p := C.ggml_init(params)
	if p == nil {
		return nil, errors.New("failed to initialize GGML context")
	}
	return &GgmlContext{p: p}, nil
}

func (ctx *GgmlContext) Free() {
	if ctx.p == nil {
		return
	}
	C.ggml_free(ctx.p)
}

type GgmlType C.enum_ggml_type

const (
	GGML_TYPE_F32 GgmlType = C.GGML_TYPE_F32
	GGML_TYPE_F16 GgmlType = C.GGML_TYPE_F16
	GGML_TYPE_I32 GgmlType = C.GGML_TYPE_I32
)

func (ctx *GgmlContext) NewGgmlTensor1D(t GgmlType, numElements int) (*GgmlTensor, error) {
	p := C.ggml_new_tensor_1d(ctx.p, C.enum_ggml_type(t), C.int64_t(numElements))
	if p == nil {
		return nil, errors.New("failed to create GGML tensor")
	}
	return &GgmlTensor{p: p}, nil
}

func (ctx *GgmlContext) NewGgmlTensor2D(t GgmlType, n0, n1 int) (*GgmlTensor, error) {
	p := C.ggml_new_tensor_2d(ctx.p, C.enum_ggml_type(t), C.int64_t(n0), C.int64_t(n1))
	if p == nil {
		return nil, errors.New("failed to create GGML tensor")
	}
	return &GgmlTensor{p: p}, nil
}

func (t *GgmlTensor) GgmlSetParam(ctx *GgmlContext) {
	C.ggml_set_param(ctx.p, t.p)
}

func (t *GgmlTensor) GgmlSetZero() {
	C.ggml_set_zero(t.p)
}

func (t *GgmlTensor) GgmlSetInput() {
	C.ggml_set_input(t.p)
}

func (t *GgmlTensor) GgmlSetOutput() {
	C.ggml_set_output(t.p)
}

// Dup
func (ctx *GgmlContext) GgmlDup(src *GgmlTensor) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_dup_tensor(ctx.p, src.p)}
}

// Transpose
func (ctx *GgmlContext) GgmlTranspose(t *GgmlTensor) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_transpose(ctx.p, t.p)}
}

// GetRows extracts the rows of a tensor
func (ctx *GgmlContext) GetRows(t *GgmlTensor, rows *GgmlTensor) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_get_rows(ctx.p, t.p, rows.p)}
}

// GgmlRMSNorm
func (ctx *GgmlContext) GgmlRMSNorm(t *GgmlTensor, eps float32) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_rms_norm(ctx.p, t.p, C.float(eps))}
}

func (ctx *GgmlContext) GgmlMul(a *GgmlTensor, b *GgmlTensor) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_mul(ctx.p, a.p, b.p)}
}

func (ctx *GgmlContext) GgmlSilu(a *GgmlTensor) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_silu(ctx.p, a.p)}
}

//	func (ctx *GgmlContext) GgmlCont(a *GgmlTensor) *GgmlTensor {
//		return &GgmlTensor{p: C.ggml_cont(ctx.p, a.p)}
//	}
func (ctx *GgmlContext) GgmlCont_2d(a *GgmlTensor, ne0 int, ne1 int) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_cont_2d(ctx.p, a.p, C.int64_t(ne0), C.int64_t(ne1))}
}
func (ctx *GgmlContext) GgmlCont_3d(a *GgmlTensor, ne0, ne1, ne2 int) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_cont_3d(ctx.p, a.p, C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2))}
}
func (ctx *GgmlContext) GgmlPermute(a *GgmlTensor, axis0 int, axis1 int, axis2 int, axis3 int) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_permute(ctx.p, a.p, C.int(axis0), C.int(axis1), C.int(axis2), C.int(axis3))}
}

func (ctx *GgmlContext) GgmlCopy(a *GgmlTensor, b *GgmlTensor) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_cpy(ctx.p, a.p, b.p)}
}

func (ctx *GgmlContext) GgmlView_3d(a *GgmlTensor, ne0, ne1, ne2 int, nb1, nb2 int64, offset int64) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_view_3d(ctx.p, a.p, C.int64_t(ne0), C.int64_t(ne1), C.int64_t(ne2), C.size_t(nb1), C.size_t(nb2), C.size_t(offset))}
}
func (ctx *GgmlContext) GgmlView_2d(a *GgmlTensor, ne0, ne1 int, nb1 int64, offset int64) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_view_2d(ctx.p, a.p, C.int64_t(ne0), C.int64_t(ne1), C.size_t(nb1), C.size_t(offset))}
}

// func (ctx *GgmlContext) GgmlSoftMax(a *GgmlTensor) *GgmlTensor {
// 	return &GgmlTensor{p: C.ggml_soft_max(ctx.p, a.p)}
// }

func (ctx *GgmlContext) GgmlSoftMax(a *GgmlTensor, mask *GgmlTensor, scale float32, maxBias float32) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_soft_max_ext(ctx.p, a.p, mask.p, C.float(scale), C.float(maxBias))}
}

func (ctx *GgmlContext) GgmlMulMat(a *GgmlTensor, b *GgmlTensor) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_mul_mat(ctx.p, a.p, b.p)}
}

func (ctx *GgmlContext) GgmlReshape_3d(a *GgmlTensor, rows int, cols int, depth int) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_reshape_3d(ctx.p, a.p, C.int64_t(rows), C.int64_t(cols), C.int64_t(depth))}
}

func (ctx *GgmlContext) GgmlReshape_2d(a *GgmlTensor, rows int, cols int) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_reshape_2d(ctx.p, a.p, C.int64_t(rows), C.int64_t(cols))}
}

func (ctx *GgmlContext) GgmlView_1d(a *GgmlTensor, ne0 int, offset int64) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_view_1d(ctx.p, a.p, C.int64_t(ne0), C.size_t(offset))}
}

// GGMLRope
func (ctx *GgmlContext) GgmlRope(a *GgmlTensor, b *GgmlTensor, c *GgmlTensor,
	n_dims int,
	mode int,
	n_ctx_orig int,
	freq_base float32,
	freq_scale float32,
	ext_factor float32,
	attn_factor float32,
	beta_fast float32,
	beta_slow float32) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_rope_ext(ctx.p, a.p, b.p, c.p,
		C.int(n_dims),
		C.int(mode),
		C.int(n_ctx_orig),
		C.float(freq_base),
		C.float(freq_scale),
		C.float(ext_factor),
		C.float(attn_factor),
		C.float(beta_fast),
		C.float(beta_slow))}
}

func (ctx *GgmlContext) GgmlAdd(a *GgmlTensor, b *GgmlTensor) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_add(ctx.p, a.p, b.p)}
}

type GgufContext struct {
	p *C.struct_gguf_context
}

type Graph struct {
	p *C.struct_ggml_cgraph
}

func (ctx *GgmlContext) NewGgmlCGraph() (*Graph, error) {
	p := C.ggml_new_graph(ctx.p)
	if p == nil {
		return nil, errors.New("failed to create GGML graph")
	}
	return &Graph{p: p}, nil
}

func (g *Graph) Free() {
	// if g.p == nil {
	// 	return
	// }
	// C.ggml_free_graph(g.p)
}

func (g *Graph) BuildForwardExpand(f *GgmlTensor) {
	C.ggml_build_forward_expand(g.p, f.p)
}

func (g *Graph) ComputeWithCtx(ctx *GgmlContext, nThreads int) {
	C.ggml_graph_compute_with_ctx(ctx.p, g.p, C.int(nThreads))
}

func (ctx *GgmlContext) GgufInitFromFile(fname string, alloc bool) (*GgufContext, error) {
	if ctx.p == nil {
		return nil, errors.New("GGML context not initialized")
	}
	params := C.struct_gguf_init_params{
		no_alloc: false, //C.bool(!alloc),
		ctx:      &ctx.p,
	}
	fname_cstr := C.CString(fname)
	defer C.free(unsafe.Pointer(fname_cstr))

	p := C.gguf_init_from_file(fname_cstr, params)
	if p == nil {
		return nil, errors.New("failed to initialize GGUF context")
	}
	return &GgufContext{p: p}, nil
}

func (ctx *GgufContext) Free() {
	if ctx.p == nil {
		return
	}
	C.gguf_free(ctx.p)
	ctx.p = nil
}

func (ctx *GgufContext) GetAlignment() int {
	return int(C.gguf_get_alignment(ctx.p))
}

func (ctx *GgufContext) GetDataOffset() int {
	return int(C.gguf_get_data_offset(ctx.p))
}

func (ctx *GgufContext) GetNKV() int {
	return int(C.gguf_get_n_kv(ctx.p))
}

func (ctx *GgufContext) GetNTensors() int {
	return int(C.gguf_get_n_tensors(ctx.p))
}

func (ctx *GgufContext) GetTensorOffset(tensorId int) int {
	return int(C.gguf_get_tensor_offset(ctx.p, C.int64_t(tensorId)))
}

func (ctx *GgufContext) GetTensorName(tensorId int) string {
	return C.GoString(C.gguf_get_tensor_name(ctx.p, C.int64_t(tensorId)))
}

func (ctx *GgufContext) GetTensorType(tensorId int) int {
	return int(C.gguf_get_tensor_type(ctx.p, C.int64_t(tensorId)))
}

func (ctx *GgufContext) GetTensorSize(tensorId int) int {
	return int(C.gguf_get_tensor_size(ctx.p, C.int64_t(tensorId)))
}

// const char * gguf_get_key(const struct gguf_context * ctx, int64_t key_id) {
//     GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
//     return ctx->kv[key_id].get_key().c_str();
// }

// enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int64_t key_id) {
//     GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
//     return ctx->kv[key_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[key_id].get_type();
// }

func (ctx *GgufContext) GetKey(keyId int) string {
	return C.GoString(C.gguf_get_key(ctx.p, C.int64_t(keyId)))
}

// GGML_API enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int64_t key_id);

// func (ctx *GgufContext) GetKeyType(keyId int) GgufType {
// 	return GgufType(C.gguf_get_kv_type(ctx.p, C.int64_t(keyId)))
// }

// // GGML_API enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int64_t key_id);

// func (ctx *GgufContext) GetArrType(keyId int) GgufType {
// 	return GgufType(C.gguf_get_arr_type(ctx.p, C.int64_t(keyId)))
// }

func (ctx *GgufContext) GetValue(keyId int) (any, error) {
	keyType := C.gguf_get_kv_type(ctx.p, C.int64_t(keyId))

	switch keyType {
	case C.GGUF_TYPE_UINT8:
		v := C.gguf_get_val_u8(ctx.p, C.int64_t(keyId))
		return int(v), nil
	case C.GGUF_TYPE_INT8:
		v := C.gguf_get_val_i8(ctx.p, C.int64_t(keyId))
		return int(v), nil
	case C.GGUF_TYPE_UINT16:
		v := C.gguf_get_val_u16(ctx.p, C.int64_t(keyId))
		return int(v), nil
	case C.GGUF_TYPE_INT16:
		v := C.gguf_get_val_i16(ctx.p, C.int64_t(keyId))
		return int(v), nil
	case C.GGUF_TYPE_UINT32:
		v := C.gguf_get_val_u32(ctx.p, C.int64_t(keyId))
		return int(v), nil
	case C.GGUF_TYPE_INT32:
		v := C.gguf_get_val_i32(ctx.p, C.int64_t(keyId))
		return int(v), nil
	case C.GGUF_TYPE_FLOAT32:
		v := C.gguf_get_val_f32(ctx.p, C.int64_t(keyId))
		return float32(v), nil
	case C.GGUF_TYPE_BOOL:
		v := C.gguf_get_val_bool(ctx.p, C.int64_t(keyId))
		return bool(v), nil
	case C.GGUF_TYPE_STRING:
		cstr := C.gguf_get_val_str(ctx.p, C.int64_t(keyId))
		return C.GoString(cstr), nil
	case C.GGUF_TYPE_UINT64:
		v := C.gguf_get_val_u64(ctx.p, C.int64_t(keyId))
		return int(v), nil
	case C.GGUF_TYPE_INT64:
		v := C.gguf_get_val_i64(ctx.p, C.int64_t(keyId))
		return int(v), nil
	case C.GGUF_TYPE_FLOAT64:
		v := C.gguf_get_val_f64(ctx.p, C.int64_t(keyId))
		return float64(v), nil
	case C.GGUF_TYPE_ARRAY:
		elementType := C.gguf_get_arr_type(ctx.p, C.int64_t(keyId))
		elementCount := C.gguf_get_arr_n(ctx.p, C.int64_t(keyId))
		switch elementType {
		case C.GGUF_TYPE_STRING:
			elements := make([]string, elementCount)
			for i := C.size_t(0); i < elementCount; i++ {
				elements[i] = C.GoString(C.gguf_get_arr_str(ctx.p, C.int64_t(keyId), i))
			}
			return elements, nil
		case C.GGUF_TYPE_INT32:
			data := C.gguf_get_arr_data(ctx.p, C.int64_t(keyId))
			p := (*C.int32_t)(data)
			elements := make([]int32, elementCount)
			for i := C.size_t(0); i < elementCount; i++ {
				elements[i] = int32(*p)
				p = (*C.int32_t)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + 4))
			}
			return elements, nil

			// case C.GGUF_TYPE_UINT8:
		// 	v := C.gguf_get_val_u8(ctx.p, C.int64_t(keyId))
		// 	return int(v), nil
		// case C.GGUF_TYPE_INT8:
		// 	v := C.gguf_get_val_i8(ctx.p, C.int64_t(keyId))
		// 	return int(v), nil
		default:
			return nil, fmt.Errorf("unknown array element type: %d", elementType)
		}
	}
	return nil, fmt.Errorf("unknown key type: %d", keyType)
}

const LLAMA_ROPE_TYPE_NORM int = int(C.LLAMA_ROPE_TYPE_NORM)

// type GgufType C.enum_gguf_type

// const (
// 	GGUF_TYPE_UINT8   GgufType = C.GGUF_TYPE_UINT8
// 	GGUF_TYPE_INT8    GgufType = C.GGUF_TYPE_INT8
// 	GGUF_TYPE_UINT16  GgufType = C.GGUF_TYPE_UINT16
// 	GGUF_TYPE_INT16   GgufType = C.GGUF_TYPE_INT16
// 	GGUF_TYPE_UINT32  GgufType = C.GGUF_TYPE_UINT32
// 	GGUF_TYPE_INT32   GgufType = C.GGUF_TYPE_INT32
// 	GGUF_TYPE_FLOAT32 GgufType = C.GGUF_TYPE_FLOAT32
// 	GGUF_TYPE_BOOL    GgufType = C.GGUF_TYPE_BOOL
// 	GGUF_TYPE_STRING  GgufType = C.GGUF_TYPE_STRING
// 	GGUF_TYPE_ARRAY   GgufType = C.GGUF_TYPE_ARRAY
// 	GGUF_TYPE_UINT64  GgufType = C.GGUF_TYPE_UINT64
// 	GGUF_TYPE_INT64   GgufType = C.GGUF_TYPE_INT64
// 	GGUF_TYPE_FLOAT64 GgufType = C.GGUF_TYPE_FLOAT64
// 	// GGUF_TYPE_COUNT   GgufType = C.GGUF_TYPE_COUNT
// )

//    // Save tensors data offset of the main file.
//     // For subsidiary files, `meta` tensor data offset must not be used,
//     // so we build a unified tensors index for weights.
//     for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
//         std::string tensor_name = std::string(cur->name);
//         // make sure there is no duplicated tensor names
//         if (weights_map.find(tensor_name) != weights_map.end()) {
//             throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
//         }
//         n_elements += ggml_nelements(cur);
//         n_bytes    += ggml_nbytes(cur);
//         weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, meta.get(), cur));
//     }
//     uint16_t n_split = 0;
//     get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

type GgmlTensor struct {
	p *C.struct_ggml_tensor
}

func ggml_ne_string(t *C.struct_ggml_tensor) string {
	var s strings.Builder
	for i := 0; i < 4; i++ {
		if i != 0 {
			s.WriteString(", ")
		}
		s.WriteString(fmt.Sprintf("%d", t.ne[i]))
	}
	return s.String()
}

func ggml_type_name(t C.enum_ggml_type) string {
	return C.GoString(C.ggml_type_name(t))
}

func ggml_op_desc(t *C.struct_ggml_tensor) string {
	return C.GoString(C.ggml_op_desc(t))
}

func (t *GgmlTensor) DebugString() string {
	var s strings.Builder

	fmt.Fprintf(&s, "%s{%s}",
		t.GetName(),
		ggml_ne_string(t.p))

	// if t.p.op == C.GGML_OP_VIEW {
	// 	offset := *((int64_t *) t.p.op_params)
	// 	fmt.Fprintf(&s, "offset = %d\n", offset)
	// }
	return s.String()
}

func (t *GgmlTensor) DebugStringWithSource() string {
	var s strings.Builder

	src0 := ""
	src1 := ""
	if t.p.src[0] != nil {
		name := C.GoString(C.ggml_get_name(t.p.src[0]))
		src0 = name + "{" + ggml_ne_string(t.p.src[0]) + "}"
	}
	if t.p.src[1] != nil {
		name := C.GoString(C.ggml_get_name(t.p.src[1]))
		src1 = name + "{" + ggml_ne_string(t.p.src[1]) + "}"
	}

	fmt.Fprintf(&s, "Tensor: %s = (%s) %10s(%s, %s) = {%s}",
		t.GetName(),
		ggml_type_name(t.p._type),
		ggml_op_desc(t.p),
		src0,
		src1,
		ggml_ne_string(t.p))

	// if t.p.op == C.GGML_OP_VIEW {
	// 	offset := *((int64_t *) t.p.op_params)
	// 	fmt.Fprintf(&s, "offset = %d\n", offset)
	// }
	return s.String()
}

func (t *GgmlTensor) Shape() []int {
	ndims := t.GetNDims()
	dims := make([]int, ndims)
	for i := 0; i < ndims; i++ {
		dims[i] = int(t.p.ne[i])
	}
	return dims
}

// print the data in the same format as the ggml_print_tensor function from the eval tool

func (t *GgmlTensor) Dump() string {
	var s strings.Builder

	s.WriteString(t.DebugString())
	s.WriteString("\n")
	canDump := true
	switch t.p._type {
	case C.GGML_TYPE_Q4_K:
		canDump = false
		fmt.Fprintf(&s, "<cannot dump data of type Q4_K>\n")
	case C.GGML_TYPE_Q5_K:
		canDump = false
		fmt.Fprintf(&s, "<cannot dump data of type Q5_K>\n")
	case C.GGML_TYPE_Q6_K:
		canDump = false
		fmt.Fprintf(&s, "<cannot dump data of type Q6_K>\n")
	}

	if !canDump {
	} else if t.p.data == nil {
		fmt.Fprintf(&s, "NIL DATA\n")
	} else {
		// static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
		// GGML_ASSERT(n > 0);
		n := C.int64_t(3)
		ne := t.p.ne
		nb := t.p.nb
		sum := float32(0)
		for i3 := C.int64_t(0); i3 < ne[3]; i3++ {
			fmt.Fprintf(&s, "                                     [\n")
			for i2 := C.int64_t(0); i2 < ne[2]; i2++ {
				if i2 == n && ne[2] > 2*n {
					fmt.Fprintf(&s, "                                      ..., \n")
					i2 = ne[2] - n
				}
				fmt.Fprintf(&s, "                                      [\n")
				for i1 := C.int64_t(0); i1 < ne[1]; i1++ {
					if i1 == n && ne[1] > 2*n {
						fmt.Fprintf(&s, "                                       ..., \n")
						i1 = ne[1] - n
					}
					fmt.Fprintf(&s, "                                       [")
					for i0 := C.int64_t(0); i0 < ne[0]; i0++ {
						if i0 == n && ne[0] > 2*n {
							fmt.Fprintf(&s, "..., ")
							i0 = ne[0] - n
						}
						i := i3*C.int64_t(nb[3]) + i2*C.int64_t(nb[2]) + i1*C.int64_t(nb[1]) + i0*C.int64_t(nb[0])
						v := float32(0)
						p := unsafe.Pointer(uintptr(t.p.data) + uintptr(i))
						switch t.p._type {
						case C.GGML_TYPE_F16:
							p := (*C.ggml_fp16_t)(p)
							v = float32(C.ggml_fp16_to_fp32(*p))
						case C.GGML_TYPE_F32:
							p := (*C.float)(p)
							v = float32(*p)
						case C.GGML_TYPE_I32:
							p := (*C.int32_t)(p)
							v = float32(*p)
						default:
							panic(fmt.Sprintf("unknown tensor type: %d", t.p._type))
						}

						// fmt.Fprintf(os.Stderr, "%d %d %d %d\n", i0, i1, i2, i3)
						// v := float32(C.ggml_get_f32_nd(t.p, C.int(i0), C.int(i1), C.int(i2), C.int(i3)))

						// if (type == GGML_TYPE_F16) {
						// 	v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
						// } else if (type == GGML_TYPE_F32) {
						// 	v = *(float *) &data[i];
						// } else if (type == GGML_TYPE_I32) {
						// 	v = (float) *(int32_t *) &data[i];
						// } else if (type == GGML_TYPE_I16) {
						// 	v = (float) *(int16_t *) &data[i];
						// } else if (type == GGML_TYPE_I8) {
						// 	v = (float) *(int8_t *) &data[i];
						// } else {
						// 	GGML_ABORT("fatal error");
						// }
						fmt.Fprintf(&s, "%12.4f", v)
						sum += v
						if i0 < ne[0]-1 {
							fmt.Fprintf(&s, ", ")
						}
					}
					fmt.Fprintf(&s, "],\n")
				}
				fmt.Fprintf(&s, "                                      ],\n")
			}
			fmt.Fprintf(&s, "                                     ]\n")
			fmt.Fprintf(&s, "                                     sum = %f\n", sum)
			nbytes := int(t.GetNbytes())
			fmt.Fprintf(&s, "                                     hash = %x\n", simpleHash(unsafe.Slice((*uint8)(t.p.data), nbytes)))
		}
	}

	return s.String()
}

// GGML_API int64_t ggml_nelements (const struct ggml_tensor * tensor);
// GGML_API int64_t ggml_nrows     (const struct ggml_tensor * tensor);
// GGML_API size_t  ggml_nbytes    (const struct ggml_tensor * tensor);
// GGML_API size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN

func (t *GgmlTensor) GetNelements() int64 {
	return int64(C.ggml_nelements(t.p))
}

func (t *GgmlTensor) GetNrows() int64 {
	return int64(C.ggml_nrows(t.p))
}

func (t *GgmlTensor) GetNbytes() int64 {
	return int64(C.ggml_nbytes(t.p))
}

func (t *GgmlTensor) GetNbytesPad() int64 {
	return int64(C.ggml_nbytes_pad(t.p))
}

// TensorType
func (t *GgmlTensor) GetTensorType() int {
	return int(t.p._type)
}

// DataFloat32
func (t *GgmlTensor) DataFloat32() []float32 {
	n := t.GetNelements()
	data := C.ggml_get_data_f32(t.p)
	return unsafe.Slice((*float32)(data), n)
}

// GGML_API int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars

func (t *GgmlTensor) GetNDims() int {
	return int(C.ggml_n_dims(t.p))
}

func (t *GgmlTensor) GetDim(i int) int {
	return int(t.p.ne[i])
}

func (t *GgmlTensor) IsScalar() bool {
	return bool(C.ggml_is_scalar(t.p))
}

func (t *GgmlTensor) IsVector() bool {
	return bool(C.ggml_is_vector(t.p))
}

func (t *GgmlTensor) IsMatrix() bool {
	return bool(C.ggml_is_matrix(t.p))
}

func (t *GgmlTensor) SetF32(value float32) {
	C.ggml_set_f32(t.p, C.float(value))
}
func (t *GgmlTensor) GetF32_1D(index int) float32 {
	return float32(C.ggml_get_f32_1d(t.p, C.int(index)))
}

func (t *GgmlTensor) SetI32_1D(index int, value int32) {
	C.ggml_set_i32_1d(t.p, C.int(index), C.int32_t(value))
}
func (t *GgmlTensor) SetF32_2D(i0, i1 int, value float32) {
	C.ggml_set_f32_nd(t.p, C.int(i0), C.int(i1), 0, 0, C.float(value))
}

func (ctx *GgmlContext) GetFirstTensor() (*GgmlTensor, error) {
	p := C.ggml_get_first_tensor(ctx.p)
	if p == nil {
		return nil, errors.New("failed to get first tensor")
	}
	return &GgmlTensor{p: p}, nil
}

func (ctx *GgmlContext) GetNextTensor(previous *GgmlTensor) (*GgmlTensor, error) {
	p := C.ggml_get_next_tensor(ctx.p, previous.p)
	if p == nil {
		return nil, nil
	}
	return &GgmlTensor{p: p}, nil
}

func (t *GgmlTensor) GetName() string {
	return C.GoString(C.ggml_get_name(t.p))
}

func (t *GgmlTensor) SetName(name string) {
	name_c := C.CString(name)
	defer C.free(unsafe.Pointer(name_c))
	C.ggml_set_name(t.p, name_c)
}

func (t *GgmlTensor) SetData(data []byte) {
	t.p.data = unsafe.Pointer(&data[0])
}

// 		        // note: this op tends to require high floating point range
// //       while for some models F16 is enough, for others it is not, so we default to F32 here
// ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

type GgmlPrecision C.enum_ggml_prec

const GGML_PREC_F32 GgmlPrecision = C.GGML_PREC_F32

func (t *GgmlTensor) SetMulMatPrecision(prec GgmlPrecision) {
	C.ggml_mul_mat_set_prec(t.p, C.enum_ggml_prec(prec))
}

// #define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
func GGML_PAD(x int, n int) int {
	return (x + n - 1) & ^(n - 1)
	// return C.GGML_PAD(x, n)
}

// #define GGML_KQ_MASK_PAD 64

func GGML_KQ_MASK_PAD() int {
	return 64
}

// GGML_ROW_SIZE
func GgmlRowSize(ggmlType GgmlType, ne int) int64 {
	return int64(C.ggml_row_size(C.enum_ggml_type(ggmlType), C.int64_t(ne)))
}

var GGML_NEGATIVE_INFINITY = math.Inf(-1)

// // ggml_backend_sched_reset(lctx.sched.get());
// // ggml_backend_sched_set_eval_callback(lctx.sched.get(), lctx.cparams.cb_eval, lctx.cparams.cb_eval_user_data);

// //	func (ctx *GgmlContext) ResetSched() {
// //		C.ggml_backend_sched_reset(ctx.p)
// //		C.ggml_backend_sched_set_eval_callback(ctx.p, ctx.cparams.cb_eval, ctx.cparams.cb_eval_user_data)
// //	}
// type GgmlBackendScheduler struct {
// 	p C.ggml_backend_sched_t
// }

// // // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
// // // preferrably to run on the same backend as the buffer
// // ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

// // sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false);

// // // initialize buffers from a max size graph (optional)
// // reserve_graph = build_graph(sched, max_batch_size);

// // // manually assign nodes to a backend (optional, should not be needed in most cases)
// // struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
// // ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

// // ggml_backend_sched_reserve(sched, reserve_graph);

// // // compute
// // graph = build_graph(sched); // the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
// // for (int i = 0; i < 10; ++i) {
// //     ggml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
// // }

// // // if there are graph inputs:
// // graph = build_graph(sched); // get a new graph that is not allocated (the metadata for the old graph is freed once ggml_free is called)
// // ggml_backend_sched_reset(sched); // clear the allocation of the previous graph
// // ggml_backend_sched_alloc_graph(sched, graph); // explicitly allocate the new graph but do not execute it
// // ggml_backend_tensor_set(input_tensor, ...); // copy data to the newly allocated graph tensors
// // ggml_backend_sched_graph_compute(sched, graph); // execute the graph

// // // as an alternative to the above it is also possible to assign the inputs to a dedicated context and
// // // allocate them statically via ggml_backend_alloc_ctx_tensors

// func (ctx *GgmlContext) NewGgmlBackendScheduler(backends []*GgmlBackend, parallel bool) (*GgmlBackendScheduler, error) {
// 	// GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel);

// 	backendPtrs := make([]*C.struct_ggml_backend, len(backends))
// 	for i, backend := range backends {
// 		backendPtrs[i] = backend.p
// 	}
// 	graphSize := 2048 // C.GGML_DEFAULT_GRAPH_SIZE
// 	p := C.ggml_backend_sched_new((**C.struct_ggml_backend)(&backendPtrs[0]), nil, C.int(len(backendPtrs)), C.size_t(graphSize), C.bool(parallel))
// 	if p == nil {
// 		return nil, errors.New("failed to create backend scheduler")
// 	}
// 	return &GgmlBackendScheduler{p: p}, nil
// }

// func (s *GgmlBackendScheduler) Free() {
// 	if s.p == nil {
// 		return
// 	}
// 	C.ggml_backend_sched_free(s.p)
// 	s.p = nil
// }

// func (s *GgmlBackendScheduler) Reset() {
// 	C.ggml_backend_sched_reset(s.p)
// }

// type BackendSchedulerCallback interface {
// 	// Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
// 	// when ask == true, the scheduler wants to know if the user wants to observe this node
// 	// this allows the scheduler to batch nodes together in order to evaluate them in a single call
// 	//
// 	// when ask == false, the scheduler is passing the node tensor to the user for observation
// 	// if the user returns false, the scheduler will cancel the graph compute
// 	//
// 	// typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);

// 	OnEval(t *GgmlTensor)
// }

// //export cgo_scheduler_callback
// func cgo_scheduler_callback(pT *C.struct_ggml_tensor, ask C.bool, userData unsafe.Pointer) C.bool {
// 	t := &GgmlTensor{p: pT}

// 	callback := cgo.Handle(userData).Value().(BackendSchedulerCallback)
// 	callback.OnEval(t)
// 	return true
// }

// func (s *GgmlBackendScheduler) SetEvalCallback(callback BackendSchedulerCallback) {
// 	// // Set a callback to be called for each resulting node during graph compute
// 	// GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);

// 	h := cgo.NewHandle(callback)

// 	C.ggml_backend_sched_set_eval_callback(s.p, C.cgo_scheduler_callback, unsafe.Pointer(C.uintptr_t(h)))
// }

// type GgmlBackend struct {
// 	p *C.struct_ggml_backend
// }

// func GgmlBackendCpuInit() (*GgmlBackend, error) {
// 	p := C.ggml_backend_cpu_init()
// 	if p == nil {
// 		return nil, errors.New("failed to create CPU backend")
// 	}
// 	return &GgmlBackend{p: p}, nil
// }

type GgmlNumaStrategy C.enum_ggml_numa_strategy

const (
	GGML_NUMA_STRATEGY_DISABLED   GgmlNumaStrategy = C.GGML_NUMA_STRATEGY_DISABLED
	GGML_NUMA_STRATEGY_DISTRIBUTE GgmlNumaStrategy = C.GGML_NUMA_STRATEGY_DISTRIBUTE
	GGML_NUMA_STRATEGY_ISOLATE    GgmlNumaStrategy = C.GGML_NUMA_STRATEGY_ISOLATE
	GGML_NUMA_STRATEGY_NUMACTL    GgmlNumaStrategy = C.GGML_NUMA_STRATEGY_NUMACTL
	GGML_NUMA_STRATEGY_MIRROR     GgmlNumaStrategy = C.GGML_NUMA_STRATEGY_MIRROR
)

// enum ggml_numa_strategy {
// 	GGML_NUMA_STRATEGY_DISABLED   = 0,
// 	GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
// 	GGML_NUMA_STRATEGY_ISOLATE    = 2,
// 	GGML_NUMA_STRATEGY_NUMACTL    = 3,
// 	GGML_NUMA_STRATEGY_MIRROR     = 4,
// 	GGML_NUMA_STRATEGY_COUNT
// };

func GgmlNumaInit(numa GgmlNumaStrategy) {
	C.ggml_numa_init(C.enum_ggml_numa_strategy(numa))
}

func (t *GgmlTensor) Cast(ctx *GgmlContext, newType GgmlType) *GgmlTensor {
	return &GgmlTensor{p: C.ggml_cast(ctx.p, t.p, C.enum_ggml_type(newType))}
}
