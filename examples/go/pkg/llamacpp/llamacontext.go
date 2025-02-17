package llamacpp

// #cgo CFLAGS: -O3 -DNDEBUG -I ../../../../include -I ../../../../ggml/include
// #cgo LDFLAGS: -L ../../../../src  -L ../../../../ggml/src -l llama -l ggml -l ggml-base -l ggml-cpu -l gomp -l common -l m  -l stdc++
// #include "llama.h"
import "C"
import "fmt"

type Token int32

func NewLlamaContextParams() C.struct_llama_context_params {
	return C.llama_context_default_params()
}

func NewLlamaModelParams() C.struct_llama_model_params {
	return C.llama_model_default_params()
}

// Load the model
func LoadAllBackends() {
	C.ggml_backend_load_all()
}

// // Initialize the context
// llama_context * ctx = llama_init_from_model(model, ctx_params);
//         if (!ctx) {
//                printf("llama_init_from_model failed");
//               return 1;
//         }

func NewLlamaContext(model *LlamaModel, ctxParams C.struct_llama_context_params) (*LlamaContext, error) {
	if model == nil || model.model == nil {
		return nil, fmt.Errorf("model is nil")
	}
	ctx := C.llama_init_from_model(model.model, ctxParams)
	if ctx == nil {
		return nil, fmt.Errorf("llama_init_from_model failed")
	}
	return &LlamaContext{p: ctx}, nil
}

type LlamaContext struct {
	p *C.struct_llama_context
}

func (c *LlamaContext) Free() {
	if c.p == nil {
		return
	}
	C.llama_free(c.p)
	c.p = nil
}

// cout << "llama_n_ctx(ctx): " << llama_n_ctx(ctx) << endl;

func (c *LlamaContext) NContext() int {
	return int(C.llama_n_ctx(c.p))
}

func (c *LlamaContext) Decode(batch *LlamaBatch) error {
	if c.p == nil || batch == nil {
		return fmt.Errorf("context or batch is nil")
	}
	if C.llama_decode(c.p, batch.c) != 0 {
		return fmt.Errorf("failed to decode")
	}
	return nil
}

func (c *LlamaContext) Sample(sampler *LlamaSampler, token Token) Token {
	return Token(C.llama_sampler_sample(sampler.sampler, c.p, C.int(token)))
}
