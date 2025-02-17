package main

import (
	"context"
	"fmt"
	"math"
	"os"

	"github.com/ggerganov/llama.cpp/examples/go/pkg/llamacpp"
)

func main() {
	ctx := context.Background()

	if err := run(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
	// if err := run_inference_at_high_level(ctx); err != nil {
	// 	fmt.Fprintf(os.Stderr, "Error: %s\n", err)
	// 	os.Exit(1)
	// }

}

func run_simple_graph(ctx context.Context) error {

	// llamacpp.GgmlNumaInit(llamacpp.GGML_NUMA_STRATEGY_NUMACTL)

	params := llamacpp.NewGgmlInitParams(16 * 1024 * 1024)
	ggmlContext, err := llamacpp.NewGgmlContext(params)
	if err != nil {
		return fmt.Errorf("failed to create load context: %w", err)
	}
	defer ggmlContext.Free()

	x, err := ggmlContext.NewGgmlTensor1D(llamacpp.GGML_TYPE_F32, 1)
	if err != nil {
		return fmt.Errorf("failed to create GGML tensor: %w", err)
	}

	x.GgmlSetParam(ggmlContext) // x is an input variable

	a, err := ggmlContext.NewGgmlTensor1D(llamacpp.GGML_TYPE_F32, 1)
	if err != nil {
		return fmt.Errorf("failed to create GGML tensor: %w", err)
	}
	b, err := ggmlContext.NewGgmlTensor1D(llamacpp.GGML_TYPE_F32, 1)
	if err != nil {
		return fmt.Errorf("failed to create GGML tensor: %w", err)
	}
	x2 := ggmlContext.GgmlMul(x, x)
	f := ggmlContext.GgmlAdd(ggmlContext.GgmlMul(a, x2), b)

	graph, err := ggmlContext.NewGgmlCGraph()
	if err != nil {
		return fmt.Errorf("failed to create graph: %w", err)
	}
	defer graph.Free()
	graph.BuildForwardExpand(f)
	//       // set the input variable and parameter values
	x.SetF32(2.0)
	a.SetF32(2.5)
	b.SetF32(0.3)

	graph.ComputeWithCtx(ggmlContext, 1)

	fmt.Printf("f = %f\n", f.GetF32_1D(0))
	return nil
}

func run(ctx context.Context) error {
	// llamacpp.GgmlNumaInit(llamacpp.GGML_NUMA_STRATEGY_DISTRIBUTE)

	// runtime.LockOSThread()
	// defer runtime.UnlockOSThread()

	modelPath := "/home/justinsb/ai/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
	// modelFile, err := os.Open(modelPath)
	// defer modelFile.Close()

	model, err := NewModelData(modelPath)
	if err != nil {
		return fmt.Errorf("failed to create model: %w", err)
	}
	defer model.Free()

	var prompt []llamacpp.Token
	tokenizer, err := NewTokenizer(modelPath)
	if err != nil {
		return fmt.Errorf("failed to create tokenizer: %w", err)
	}
	{

		promptString := `<|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024
<|eot_id|>
What is today?
`

		tokens, err := tokenizer.Tokenize(ctx, promptString)
		if err != nil {
			return fmt.Errorf("failed to tokenize: %w", err)
		}
		fmt.Println(tokens)
		prompt = tokens
	}

	input := prompt

	for {
		result_output, err := computeNextToken(ctx, model, input)
		if err != nil {
			return fmt.Errorf("failed to compute next token: %w", err)
		}
		fmt.Println(result_output.DebugString())

		// Find top prediction

		{
			// model, err := llamacpp.NewLlamaModel(modelPath, llamacpp.NewLlamaModelParams())
			// if err != nil {
			// 	return fmt.Errorf("failed to create model: %w", err)
			// }
			// defer model.Free()

			// vocab, err := model.GetVocab()
			logits := make([]float32, result_output.GetDim(0))
			for i := 0; i < len(logits); i++ {
				logits[i] = result_output.GetF32_1D(i)
			}

			max_logit := logits[0]
			max_logit_index := 0
			for i := 1; i < len(logits); i++ {
				if logits[i] > max_logit {
					max_logit = logits[i]
					max_logit_index = i
				}
			}

			fmt.Printf("greedy logit sampler: %v, %v\n", max_logit_index, logits[max_logit_index])

			nextToken := llamacpp.Token(max_logit_index)
			s := model.TokenToString(nextToken)
			fmt.Printf("token: %q\n", s)

			if model.IsEOG(nextToken) {
				break
			}
			input = append(input, nextToken)
			// piece, err := vocab.ConvertTokenToString(llamacpp.Token(max_logit_index))
			// if err != nil {
			// 	return fmt.Errorf("failed to convert token to string: %w", err)
			// }
			// fmt.Printf("piece: %v\n", piece)
		}
		// sampler := llamacpp.NewLlamaSampler(llamacpp.LlamaSamplerOptions{
		// 	MinP: &llamacpp.MinPOptions{
		// 		P:       0.05,
		// 		MinKeep: 1,
		// 	},
		// })
		// defer sampler.Free()

		//   // sample the next token, check is it an end of generation?
		//   new_token_id := sampler.Sample(ctx, -1);
		//   if (llama_vocab_is_eog(vocab, new_token_id)) {
		// 	  break;
		//   }

		//   std::string piece;
		//   if (convert_token_to_string(vocab, new_token_id, piece)) {
		// 	  return 1;
		//   }

	}

	return nil
}

type ModelData struct {
	gguf        *llamacpp.GgufContext
	ggmlContext *llamacpp.GgmlContext
	metadata    map[string]any
	tensors     map[string]*llamacpp.GgmlTensor
	vocab       []string
	eog_tokens  map[llamacpp.Token]struct{}
}

func (m *ModelData) VocabTokenCount() int {
	return len(m.vocab)
}

func (m *ModelData) TokenToString(token llamacpp.Token) string {
	return m.vocab[token]
}

func (m *ModelData) IsEOG(token llamacpp.Token) bool {
	_, found := m.eog_tokens[token]
	return found
}

func (m *ModelData) Free() {
	m.gguf.Free()
	m.ggmlContext.Free()
	// for _, tensor := range m.tensors {
	// 	tensor.Free()
	// }
}

func NewModelData(modelPath string) (*ModelData, error) {
	params := llamacpp.NewGgmlInitParams(32 * 1024 * 1024)
	ggmlContext, err := llamacpp.NewGgmlContext(params)
	if err != nil {
		return nil, fmt.Errorf("failed to create load context: %w", err)
	}

	withAlloc := true
	gguf, err := ggmlContext.GgufInitFromFile(modelPath, withAlloc)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize GGUF context: %w", err)
	}
	// TODO: We can probably gguf free here ...
	// defer gguf.Free()

	// fmt.Println(gguf.GetAlignment())
	// fmt.Println(gguf.GetDataOffset())
	// fmt.Println(gguf.GetNKV())

	metadata := make(map[string]any)
	for i := 0; i < gguf.GetNKV(); i++ {
		key := gguf.GetKey(i)
		value, err := gguf.GetValue(i)
		if err != nil {
			return nil, fmt.Errorf("failed to get value for %q: %w", key, err)
		}
		metadata[key] = value
	}

	// tensorMeta := make(map[string]*TensorMeta)

	// for i := 0; i < gguf.GetNTensors(); i++ {
	// 	// fmt.Println("--------------------------------")
	// 	// fmt.Printf("tensor #%d\n", i)
	// 	// fmt.Printf("offset: %d\n", gguf.GetTensorOffset(i))
	// 	// fmt.Printf("name: %s\n", gguf.GetTensorName(i))
	// 	// fmt.Printf("type: %d\n", gguf.GetTensorType(i))
	// 	// fmt.Printf("size: %d\n", gguf.GetTensorSize(i))
	// 	tensorMeta[gguf.GetTensorName(i)] = &TensorMeta{
	// 		offset:     gguf.GetTensorOffset(i),
	// 		size:       gguf.GetTensorSize(i),
	// 		tensorType: gguf.GetTensorType(i),
	// 	}
	// }

	tensors := make(map[string]*llamacpp.GgmlTensor)

	tensor, err := ggmlContext.GetFirstTensor()
	if err != nil {
		return nil, fmt.Errorf("failed to get first tensor: %w", err)
	}
	tensors[tensor.GetName()] = tensor
	for {
		tensor, err = ggmlContext.GetNextTensor(tensor)
		if tensor == nil {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to get next tensor: %w", err)
		}

		// fmt.Println(tensor.GetName())
		tensors[tensor.GetName()] = tensor
	}

	vocab := metadata["tokenizer.ggml.tokens"].([]string)
	// tokenTypes := metadata["tokenizer.ggml.token_type"].([]int32)

	eog_tokens := make(map[llamacpp.Token]struct{})
	if v, ok := metadata["tokenizer.ggml.eos_token_id"]; ok {
		eog_tokens[llamacpp.Token(v.(int))] = struct{}{}
	}
	// for i, tokenType := range tokenTypes {
	// 	if tokenType == 1 {
	// 		eog_tokens[i] = struct{}{}
	// 	}
	// }
	return &ModelData{
		gguf:        gguf,
		ggmlContext: ggmlContext,
		metadata:    metadata,
		tensors:     tensors,
		vocab:       vocab,
		eog_tokens:  eog_tokens,
	}, nil
}

func computeNextToken(ctx context.Context, model *ModelData, input []llamacpp.Token) (*llamacpp.GgmlTensor, error) {
	numThreads := 16

	fmt.Printf("tokens: %v\n", input)
	n_tokens := len(input)

	n_rot := -1
	rope_type := -1
	freq_base := float32(-1.0)
	freq_scale := float32(-1.0)
	ext_factor := float32(-1.0)
	attn_factor := float32(-1.0)
	yarn_beta_fast := float32(32.0)
	yarn_beta_slow := float32(1.0)
	n_ctx_orig := -1
	f_attention_scale := float32(-1.0)
	f_max_alibi_bias := float32(-1.0)
	rope_attn_factor := float32(1.0)
	n_ctx_train := -1
	f_norm_rms_eps := float32(-1.0)
	n_embd := -1

	architecture := ""

	n_head_kv := -1
	n_head := -1
	n_embd_head_k := -1
	n_embd_head_v := -1
	for key, value := range model.metadata {
		switch key {
		case "general.architecture":
			architecture = value.(string)
		case "llama.rope.dimension_count":
			n_rot = value.(int)
		// case "llama.rope.type":
		// 	rope_type = value
		case "llama.rope.freq_base":
			freq_base = value.(float32)
			// case "llama.attention.layer_norm_rms_epsilon":
			// 	f_attention_scale = value
		case "llama.rope.scaling.attn_factor":
			rope_attn_factor = value.(float32)

		case "llama.context_length":
			n_ctx_train = value.(int)

		case "llama.attention.layer_norm_rms_epsilon":
			f_norm_rms_eps = value.(float32)

		case "llama.embedding_length":
			n_embd = value.(int)

			// ml.get_key(LLM_KV_CONTEXT_LENGTH,    hparams.n_ctx_train);
			// ml.get_key(LLM_KV_BLOCK_COUNT,       hparams.n_layer);
			// ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert,      false);
			// ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

			// ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

			// ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
		// ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

		// ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

		case "llama.attention.head_count":
			n_head = value.(int)
		case "llama.attention.head_count_kv":
			n_head_kv = value.(int)

		case "llama.attention.key_length":
			n_embd_head_k = value.(int)
		case "llama.attention.value_length":
			n_embd_head_v = value.(int)
		}
		// fmt.Printf("%s: %v\n", key, value)
	}

	if architecture == "" {
		return nil, fmt.Errorf("failed to get architecture")
	}

	switch architecture {
	case "llama":
		rope_type = llamacpp.LLAMA_ROPE_TYPE_NORM
	default:
		return nil, fmt.Errorf("unsupported architecture: %s", architecture)
	}

	if rope_type == -1 {
		return nil, fmt.Errorf("failed to get rope_type")
	}
	if freq_base == -1 {
		return nil, fmt.Errorf("failed to get freq_base")
	}
	beta_fast := yarn_beta_fast
	beta_slow := yarn_beta_slow
	if n_ctx_train == -1 {
		// return fmt.Errorf("failed to get n_ctx_orig")
	}
	if n_ctx_orig == -1 {
		// cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
		// hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
		// 							   hparams.n_ctx_train;
		n_ctx_orig = n_ctx_train
		// return fmt.Errorf("failed to get n_ctx_orig")
	}
	if f_attention_scale == -1 {
		f_attention_scale = 0.0
		// return fmt.Errorf("failed to get f_attention_scale")
	}
	if f_max_alibi_bias == -1 {
		f_max_alibi_bias = 0.0
		// return fmt.Errorf("failed to get f_max_alibi_bias")
	}

	if freq_scale == -1 {
		// rope_freq_scale (inverse of the kv) is optional
		ropescale := float32(0.0)
		if v, ok := model.metadata["llama.rope.scaling.factor"]; ok {
			ropescale = v.(float32)
		} else if v, ok := model.metadata["llama.rope.scale_linear"]; ok {
			ropescale = v.(float32)
		}
		if ropescale == 0.0 {
			freq_scale = 1.0
		} else {
			freq_scale = 1.0 / ropescale
		}
	}

	if ext_factor == -1 {
		// 	if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
		// 	cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
		// }
		ext_factor = 1.0
		// return fmt.Errorf("failed to get ext_factor")
	}

	if attn_factor == -1 {
		yarn_attn_factor := float32(1.0)
		yarn_attn_factor *= rope_attn_factor
		attn_factor = yarn_attn_factor
	}

	if f_norm_rms_eps == -1 {
		// f_norm_rms_eps = 1e-5
		return nil, fmt.Errorf("failed to get f_norm_rms_eps")
	}

	if n_head_kv == -1 {
		n_head_kv = n_head
	}

	if n_embd == -1 {
		return nil, fmt.Errorf("failed to get n_embd")
	}

	if n_embd_head_k == -1 {
		//     hparams.n_embd_head_k = hparams.n_embd / hparams.n_head();
		// ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);
		n_embd_head_k = n_embd / n_head
	}
	if n_embd_head_v == -1 {
		n_embd_head_v = n_embd / n_head
	}

	type TensorMeta struct {
		offset     int
		size       int
		tensorType int
	}

	// token_embd.weight
	tokenEmbedWeights := model.tensors["token_embd.weight"]
	if tokenEmbedWeights == nil {
		return nil, fmt.Errorf("failed to get token_embd.weight tensor")
	}
	// fmt.Println(tokenEmbedWeights.GetName())

	evalParams := llamacpp.NewGgmlInitParams(4 * 1024 * 1024 * 1024)
	evalContext, err := llamacpp.NewGgmlContext(evalParams)
	if err != nil {
		return nil, fmt.Errorf("failed to create eval context: %w", err)
	}

	tokens, err := evalContext.NewGgmlTensor1D(llamacpp.GGML_TYPE_I32, n_tokens)
	if err != nil {
		return nil, fmt.Errorf("failed to create GGML tensor: %w", err)
	}
	tokens.GgmlSetInput()

	inp_pos, err := evalContext.NewGgmlTensor1D(llamacpp.GGML_TYPE_I32, n_tokens)
	if err != nil {
		return nil, fmt.Errorf("failed to create inp_pos tensor: %w", err)
	}
	inp_pos.SetName("inp_pos")

	rope_freqs_weights := model.tensors["rope_freqs.weight"]
	if rope_freqs_weights == nil {
		return nil, fmt.Errorf("failed to get rope_freqs.weight tensor")
	}

	inp_embed := evalContext.GetRows(tokenEmbedWeights, tokens) //evalContext.GgmlTranspose(tokens))
	inp_embed.SetName("inp_embed")

	layer_input := inp_embed

	n_embd_head := 128

	kq_scale := f_attention_scale
	if kq_scale == 0 {
		kq_scale = float32(1.0 / math.Sqrt(float64(n_embd_head)))
	}

	KQ_mask, err := evalContext.NewGgmlTensor2D(llamacpp.GGML_TYPE_F32, n_tokens, n_tokens)
	if err != nil {
		return nil, fmt.Errorf("failed to create KQ_mask tensor: %w", err)
	}
	KQ_mask.SetName("KQ_mask")
	KQ_mask.GgmlSetInput() // maybe?

	n_outputs := 1
	n_layers := 32

	inp_out_ids, err := evalContext.NewGgmlTensor1D(llamacpp.GGML_TYPE_I32, n_outputs)
	if err != nil {
		return nil, fmt.Errorf("failed to create inp_out_ids tensor: %w", err)
	}
	inp_out_ids.SetName("inp_out_ids")
	inp_out_ids.GgmlSetInput()

	type Cache struct {
		k *llamacpp.GgmlTensor
		v *llamacpp.GgmlTensor
	}

	// caches := make([]*Cache, n_layers)
	// for l := 0; l < n_layers; l++ {
	// 	k, err := evalContext.NewGgmlTensor1D(llamacpp.GGML_TYPE_F16, 4194304)
	// 	if err != nil {
	// 		return nil, fmt.Errorf("failed to create k tensor: %w", err)
	// 	}
	// 	k.SetName(fmt.Sprintf("cache_k_l%d", l))
	// 	v, err := evalContext.NewGgmlTensor1D(llamacpp.GGML_TYPE_F16, 4194304)
	// 	if err != nil {
	// 		return nil, fmt.Errorf("failed to create v tensor: %w", err)
	// 	}
	// 	v.SetName(fmt.Sprintf("cache_v_l%d", l))
	// 	caches[l] = &Cache{
	// 		k: k,
	// 		v: v,
	// 	}
	// }

	type Layer struct {
		attn_norm_weight   *llamacpp.GgmlTensor
		attn_norm_eps      *llamacpp.GgmlTensor
		attn_output_weight *llamacpp.GgmlTensor
		ffn_norm_weight    *llamacpp.GgmlTensor
		ffn_gate_weight    *llamacpp.GgmlTensor
		ffn_up_weight      *llamacpp.GgmlTensor
		ffn_down_weight    *llamacpp.GgmlTensor
		attn_q_weight      *llamacpp.GgmlTensor
		attn_v_weight      *llamacpp.GgmlTensor
		q                  *llamacpp.GgmlTensor
		norm               *llamacpp.GgmlTensor
		attn_norm          *llamacpp.GgmlTensor
		qcur               *llamacpp.GgmlTensor
		qcur_pre           *llamacpp.GgmlTensor
		qcur_reshaped      *llamacpp.GgmlTensor

		attn_k_weight *llamacpp.GgmlTensor
		kcur_n_pre    *llamacpp.GgmlTensor
		kcur_n        *llamacpp.GgmlTensor
		k_cache_view  *llamacpp.GgmlTensor
		k_cache_copy  *llamacpp.GgmlTensor
		k             *llamacpp.GgmlTensor

		v          *llamacpp.GgmlTensor
		vcur_n_pre *llamacpp.GgmlTensor
		// vcur_n         *llamacpp.GgmlTensor
		vcur_n_transposed *llamacpp.GgmlTensor
		cache_v           *llamacpp.GgmlTensor
		v_cache_view_n    *llamacpp.GgmlTensor
		v_cache_copy      *llamacpp.GgmlTensor

		kq                *llamacpp.GgmlTensor
		kq_soft_max_ext   *llamacpp.GgmlTensor
		kqv               *llamacpp.GgmlTensor
		kqv_out_n         *llamacpp.GgmlTensor
		kqv_merged_n      *llamacpp.GgmlTensor
		kqv_merged_cont_n *llamacpp.GgmlTensor

		ffn_inp *llamacpp.GgmlTensor
		ffn_out *llamacpp.GgmlTensor

		layer_output *llamacpp.GgmlTensor
	}

	layers := make([]*Layer, n_layers)
	for l := 0; l < n_layers; l++ {
		// fmt.Printf("Building layer %d\n", l)
		layer := &Layer{}
		layers[l] = layer

		blk_n_attn_norm_weight := model.tensors[fmt.Sprintf("blk.%d.attn_norm.weight", l)]
		if blk_n_attn_norm_weight == nil {
			return nil, fmt.Errorf("failed to get blk.%d.attn_norm.weight tensor", l)
		}
		layer.attn_norm_weight = blk_n_attn_norm_weight

		blk_n_attn_output_weight := model.tensors[fmt.Sprintf("blk.%d.attn_output.weight", l)]
		if blk_n_attn_output_weight == nil {
			return nil, fmt.Errorf("failed to get blk.%d.attn_output.weight tensor", l)
		}
		layer.attn_output_weight = blk_n_attn_output_weight
		blk_n_attn_q_weight := model.tensors[fmt.Sprintf("blk.%d.attn_q.weight", l)]
		if blk_n_attn_q_weight == nil {
			return nil, fmt.Errorf("failed to get blk.%d.attn_q.weight tensor", l)
		}
		layer.attn_q_weight = blk_n_attn_q_weight

		blk_n_ffn_norm_weight := model.tensors[fmt.Sprintf("blk.%d.ffn_norm.weight", l)]
		if blk_n_ffn_norm_weight == nil {
			return nil, fmt.Errorf("failed to get blk.%d.ffn_norm.weight tensor", l)
		}
		layer.ffn_norm_weight = blk_n_ffn_norm_weight

		blk_n_ffn_gate_weight := model.tensors[fmt.Sprintf("blk.%d.ffn_gate.weight", l)]
		if blk_n_ffn_gate_weight == nil {
			return nil, fmt.Errorf("failed to get blk.%d.ffn_gate.weight tensor", l)
		}
		layer.ffn_gate_weight = blk_n_ffn_gate_weight

		blk_n_ffn_up_weight := model.tensors[fmt.Sprintf("blk.%d.ffn_up.weight", l)]
		if blk_n_ffn_up_weight == nil {
			return nil, fmt.Errorf("failed to get blk.%d.ffn_up.weight tensor", l)
		}
		layer.ffn_up_weight = blk_n_ffn_up_weight

		blk_n_ffn_down_weight := model.tensors[fmt.Sprintf("blk.%d.ffn_down.weight", l)]
		if blk_n_ffn_down_weight == nil {
			return nil, fmt.Errorf("failed to get blk.%d.ffn_down.weight tensor", l)
		}
		layer.ffn_down_weight = blk_n_ffn_down_weight

		// Attention layer
		{
			// Note that there are two nodes called norm-0 in llama.cpp - potentially confusing

			// Input layer is (4096 x n_tokens)

			// Normalize the input using RMSNorm
			// input: layer_input is (4096 x n_tokens)
			// output: norm_n is (4096 x n_tokens) (we scale each element by a constant factor)
			norm_n := evalContext.GgmlRMSNorm(layer_input, f_norm_rms_eps)
			norm_n.SetName(fmt.Sprintf("norm-%d", l))
			//fmt.Println("norm_n", norm_n.DebugString())
			layer.norm = norm_n
			fmt.Println("norm_n", norm_n.DebugString())

			// Multiply by the attention norm weight parameters
			// input: norm_n is (4096 x n_tokens)
			// input: blk_n_attn_norm_weight is (4096 x 4096)
			// Note also that we're doing a mul, not a matmul
			// output: attn_norm_n is (4096 x n_tokens)
			attn_norm_n := evalContext.GgmlMul(norm_n, blk_n_attn_norm_weight)
			fmt.Println("blk_n_attn_norm_weight", blk_n_attn_norm_weight.DebugString())
			attn_norm_n.SetName(fmt.Sprintf("attn_norm-%d", l))
			layer.attn_norm = attn_norm_n

			// Q
			{
				fmt.Println("blk_n_attn_q_weight", blk_n_attn_q_weight.DebugString()) // Shape: 4096, 4096 ; though actually a projection of 4096 x 128 x 32
				fmt.Println("attn_norm_n", attn_norm_n.DebugString())                 // Shape: 4096, n_tokens

				// blk_n_attn_q_weight = evalContext.GgmlReshape_3d(blk_n_attn_q_weight, 4096, 128, 32)

				// matmul the attn_norm_n by the attention q weight parameters
				// input: blk_n_attn_q_weight is (4096 x 4096) - though actually 4096 x 128 x 32
				// input: attn_norm_n is (4096 x n_tokens)
				// output: qcur_n_pre is (4096 x n_tokens) (TODO: is it really 128 x 32 x n_tokens?)
				fmt.Println("mulmat", blk_n_attn_q_weight.DebugString(), attn_norm_n.DebugString())
				qcur_n_pre := evalContext.GgmlMulMat(blk_n_attn_q_weight, attn_norm_n) // Shape: 4096, n_tokens (though really 128 x 32 x n_tokens)
				qcur_n_pre.SetName(fmt.Sprintf("qcur_n_pre-%d", l))
				layer.qcur_pre = qcur_n_pre // Shape: (dimensions, num_tokens, batch_size)
				qcur := qcur_n_pre

				// blk_n_attn_q_weight Tensor: blk.0.attn_q.weight = (q6_K)       NONE(, ) = {4096, 4096, 1, 1}
				// attn_norm_n Tensor: attn_norm-0 = (f32)        MUL(norm-0{4096, 31, 1, 1}, blk.0.attn_norm.weight{4096, 1, 1, 1}) = {4096, 31, 1, 1}
				// qcur_n_pre Tensor: qcur_n_pre-0 = (f32)    MUL_MAT(blk.0.attn_q.weight{4096, 4096, 1, 1}, attn_norm-0{4096, 31, 1, 1}) = {4096, 31, 1, 1}

				fmt.Println("qcur_n_pre", qcur_n_pre.DebugString())

				// Reshape => Shape is now actually 128, 32, n_tokens
				qcur_n_reshaped := evalContext.GgmlReshape_3d(qcur_n_pre, n_embd_head, n_head, n_tokens) // Shape 128, 32, n_tokens
				fmt.Println("qcur_n_reshaped", qcur_n_reshaped.DebugString())
				layer.qcur_reshaped = qcur_n_reshaped
				qcur = qcur_n_reshaped

				// Rope => Shape is still 128, 32, n_tokens
				after_rope := evalContext.GgmlRope(qcur, inp_pos, rope_freqs_weights,
					n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
					ext_factor, attn_factor, beta_fast, beta_slow)
				qcur = after_rope

				qcur = evalContext.GgmlPermute(qcur, 0, 2, 1, 3)

				// // Permute => Shape 128, n_tokens, 32
				// q_n := evalContext.GgmlPermute(qcur_n, 0, 2, 1, 3)
				// fmt.Println("q_n", q_n.DebugString())
				// q_n.SetName(fmt.Sprintf("q-%d", l))
				// layer.q = q_n

				layer.q = qcur
				fmt.Println("layer.q", layer.q.DebugString())
			}
			// Qcur_0_rope_transposed := evalContext.GgmlTranspose(qcur_n_rope)

			// Qcur_0_rope_transposed_view := evalContext.GgmlView(Qcur_0_rope_transposed, 128, 32, 512)

			// kv_head := 0 // index of where we store new KV data in the cache

			// K
			{
				// We're using Grouped Query Attention, there are 8 key value heads for the 32 heads.
				// This is a reduction factor of 4.
				// so the weights are (4096, 1024), instead of (4096, 4096)

				// 				kcur_n_pre Tensor:  = (f32)    MUL_MAT(blk.0.attn_k.weight{4096, 1024, 1, 1}, attn_norm-0{4096, 31, 1, 1}) = {1024, 31, 1, 1}
				// kcur_n_reshaped Tensor:  (reshaped) = (f32)    RESHAPE({1024, 31, 1, 1}, ) = {128, 8, 31, 1}
				// kcur_n Tensor: KCur-0 = (f32)       ROPE( (reshaped){128, 8, 31, 1}, inp_pos{31, 1, 1, 1}) = {128, 8, 31, 1}
				// k_n Tensor: KCur-0 (permuted) = (f32)    PERMUTE(KCur-0{128, 8, 31, 1}, ) = {128, 31, 8, 1}
				// k Tensor: KCur-0 (permuted) (transposed) = (f32)  TRANSPOSE(KCur-0 (permuted){128, 31, 8, 1}, ) = {31, 128, 8, 1}

				// blk.%d.attn_k.weight.  Shape: 1024, 4096 ; though actually a projection of 4096 x 128 x 8

				blk_n_attn_k_weight := model.tensors[fmt.Sprintf("blk.%d.attn_k.weight", l)]
				if blk_n_attn_k_weight == nil {
					return nil, fmt.Errorf("failed to get blk.%d.attn_k.weight tensor", l)
				}

				// blk_n_attn_k_weight = evalContext.GgmlReshape_3d(blk_n_attn_k_weight, 4096, 128, 8)

				layer.attn_k_weight = blk_n_attn_k_weight

				// kcur_n_pre = matmul(blk.0.attn_k.weight, attn_norm).  Shape: 1024, n_tokens  (though actually 128 x 8 x n_tokens)

				fmt.Println("blk_n_attn_k_weight", blk_n_attn_k_weight.DebugString()) // Shape: 4096, 1024 ; though actually a projection of 4096 x 128 x 8
				fmt.Println("attn_norm_n", attn_norm_n.DebugString())                 // Shape: 4096, n_tokens

				// matmul the attn_norm_n by the attention k weight parameters
				// input: blk_n_attn_k_weight is (4096 x 1024) - though actually 4096 x 128 x 8
				// input: attn_norm_n is (4096 x n_tokens)
				// output: kcur_n_pre is (1024 x n_tokens) (TODO: is it really 128 x 8 x n_tokens?)
				fmt.Println("mulmat", blk_n_attn_k_weight.DebugString(), attn_norm_n.DebugString())
				kcur := evalContext.GgmlMulMat(blk_n_attn_k_weight, attn_norm_n)
				fmt.Println("kcur_n_pre", kcur.DebugString())
				// layer.kcur_n_pre = kcur_n_pre

				// Reshape => Shape is now actually (128 x 8 x n_tokens)
				kcur_n_reshaped := evalContext.GgmlReshape_3d(kcur, n_embd_head, n_head_kv, n_tokens)
				fmt.Println("kcur_n_reshaped", kcur_n_reshaped.DebugString())
				kcur = kcur_n_reshaped

				// Apply rope.  Shape remains (128 x 8 x n_tokens)
				kcur = evalContext.GgmlRope(kcur, inp_pos, rope_freqs_weights,
					n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
					ext_factor, attn_factor, beta_fast, beta_slow)
				// kcur_n.SetName(fmt.Sprintf("KCur-%d", l))
				fmt.Println("kcur_n", kcur.DebugString())

				kcur = evalContext.GgmlPermute(kcur, 0, 2, 1, 3)

				// layer.kcur_n = kcur
				layer.k = kcur

				// cache_k := caches[l].k //["cache_k_l0"]
				// // fmt.Println("cache_k_ln", cache_k.DebugString())

				// // struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv.k_l[il], n_tokens*n_embd_k_gqa, ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa)*kv_head);
				// // cb(k_cache_view, "k_cache_view", il);

				// // struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv.k_l[il], n_tokens*n_embd_k_gqa, ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa)*kv_head);
				// // cb(k_cache_view, "k_cache_view", il);

				// n_embd_k_gqa := n_embd_head_k * n_head_kv

				// k_cache_view_n := evalContext.GgmlView_1d(cache_k,
				// 	n_tokens*n_embd_k_gqa,
				// 	llamacpp.GgmlRowSize(llamacpp.GGML_TYPE_F16, n_embd_k_gqa)*int64(kv_head),
				// )
				// k_cache_view_n.SetName(fmt.Sprintf("k_cache_view-%d", l))
				// // fmt.Println("view_1d", n_tokens*n_embd_k_gqa, llamacpp.GgmlRowSize(llamacpp.GGML_TYPE_F16, n_embd_k_gqa)*int64(kv_head))

				// // fmt.Println("k_cache_view_n", k_cache_view_n.DebugString())
				// layer.k_cache_view = k_cache_view_n

				// layer.k_cache_copy = evalContext.GgmlCopy(kcur_n, k_cache_view_n)

				// 	struct ggml_tensor * k =
				// 	ggml_view_3d(ctx, kv.k_l[il],
				// 			n_embd_head_k, n_kv, n_head_kv,
				// 			ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa),
				// 			ggml_row_size(kv.k_l[il]->type, n_embd_head_k),
				// 			0);
				// cb(k, "k", il);

				// k_n := evalContext.GgmlView_3d(cache_k, n_embd_head_k, n_kv, n_head_kv,
				// 	llamacpp.GgmlRowSize(llamacpp.GGML_TYPE_F16, n_embd_k_gqa),
				// 	llamacpp.GgmlRowSize(llamacpp.GGML_TYPE_F16, n_embd_head_k),
				// 	0)
				// // fmt.Println("k_n", k_n.DebugString())
				// k_n.SetName(fmt.Sprintf("k-%d", l))
				// layer.k = k_n

				// // Permute => Shape 128, n_tokens, 8
				// k_n := evalContext.GgmlPermute(kcur_n, 0, 2, 1, 3)
				// fmt.Println("k_n", k_n.DebugString())
				// k_n_t := evalContext.GgmlTranspose(kcur_n)
				// fmt.Println("k_n_t", k_n_t.DebugString())
				// // q_n.SetName(fmt.Sprintf("q-%d", l))
				// // layer.q = q_n

				// layer.k = k_n
				fmt.Println("layer.k", layer.k.DebugString())
			}

			// V
			{
				// n_ctx := 4096 // TODO: What is this value?

				// We're using Grouped Query Attention, there are 8 key value heads.
				// so the weights are (4096, 1024), instead of (4096, 4096)

				// blk.%d.attn_v.weight.  Shape: 1024, 4096 ; though actually a projection of 4096 x 128 x 8
				blk_n_attn_v_weight := model.tensors[fmt.Sprintf("blk.%d.attn_v.weight", l)]
				if blk_n_attn_v_weight == nil {
					return nil, fmt.Errorf("failed to get blk.%d.attn_v.weight tensor", l)
				}
				// blk_n_attn_v_weight = evalContext.GgmlReshape_3d(blk_n_attn_v_weight, 4096, 128, 8)
				layer.attn_v_weight = blk_n_attn_v_weight

				fmt.Println("blk_n_attn_v_weight", blk_n_attn_v_weight.DebugString()) // Shape: 4096, 1024 ; though actually a projection of 4096 x 128 x 8
				fmt.Println("attn_norm_n", attn_norm_n.DebugString())                 // Shape: 4096, n_tokens

				// 	fmt.Println("blk_n_attn_v_weight", blk_n_attn_v_weight.DebugString())
				// fmt.Println("attn_norm_n", attn_norm_n.DebugString())

				// vcur_n_pre = matmul(blk.0.attn_v.weight, attn_norm).  Shape: 1024, n_tokens  (though actually 128 x 8 x n_tokens)

				// matmul the attn_norm_n by the attention v weight parameters
				// input: blk_n_attn_v_weight is (4096 x 1024) - though actually 4096 x 128 x 8
				// input: attn_norm_n is (4096 x n_tokens)
				// output: vcur_n_pre is (1024 x n_tokens) (TODO: is it really 128 x 8 x n_tokens?)
				vcur_n_pre := evalContext.GgmlMulMat(blk_n_attn_v_weight, attn_norm_n)
				// vcur_n_reshaped := evalContext.GgmlReshape_3d(vcur_n_pre, n_embd_head, n_head, n_tokens)
				// vcur_n := evalContext.GgmlRope(vcur_n_reshaped, inp_pos, rope_freqs_weights,
				// 	n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
				// 	ext_factor, attn_factor, beta_fast, beta_slow)
				vcur_n_pre.SetName(fmt.Sprintf("Vcur-%d", l))
				layer.vcur_n_pre = vcur_n_pre
				vcur := vcur_n_pre
				fmt.Println("vcur_n_pre", vcur_n_pre.DebugString())

				// vcur = evalContext.GgmlReshape_3d(vcur, 128, 8, n_tokens) // n_embd_head*n_head_kv)

				// vcur_n_pre Vcur-25{31, 128, 8, 1}

				fmt.Println("vcur", vcur.DebugString())

				// // Reshape => Shape is now actually 128, 8, n_tokens
				// vcur_n_reshaped := evalContext.GgmlReshape_3d(vcur_n_pre, n_tokens, n_embd_head, n_head_kv)
				// fmt.Println("vcur_n_reshaped", vcur_n_reshaped.DebugString())
				// vcur = vcur_n_reshaped

				vcur = evalContext.GgmlTranspose(vcur)
				fmt.Println("vcur transpose", vcur.DebugString())

				vcur = evalContext.GgmlCont_3d(vcur, n_tokens, 128, 8)
				fmt.Println("vcur reshape", vcur.DebugString())

				// vcur = evalContext.GgmlPermute(vcur, 0, 1, 2, 3)
				// // fmt.Println("qcur_n", qcur_n.DebugString())
				// layer.v = vcur // evalContext.GgmlReshape_2d(after_rope, n_embd_head*n_head, n_tokens)

				// vcur = evalContext.GgmlReshape_3d(vcur, n_tokens, 128, 8) // n_embd_head*n_head_kv)

				layer.v = vcur

				// vcur_n_transposed := evalContext.GgmlTranspose(vcur_n_reshaped)
				// // vcur_n_transposed.SetName(fmt.Sprintf("vcur_n_transposed-%d", l))
				// layer.vcur_n_transposed = vcur_n_transposed

				// fmt.Println("vcur_n_transposed", vcur_n_transposed.DebugString())

				// cache_v := caches[l].v //["cache_k_l0"]
				// // fmt.Println("cache_v_ln", cache_v.DebugString())
				// cache_v.SetName(fmt.Sprintf("cache_v_l%d", l))
				// layer.cache_v = cache_v

				//      // note: the V cache is transposed when not using flash attention
				// 	 v_cache_view = ggml_view_2d(ctx, kv.v_l[il], n_tokens, n_embd_v_gqa,
				// 		(  n_ctx)*ggml_element_size(kv.v_l[il]),
				// 		(kv_head)*ggml_element_size(kv.v_l[il]));

				// v_cur = ggml_transpose(ctx, v_cur);

				// n_embd_v_gqa := n_embd_head_v * n_head_kv
				// // const int32_t
				// v_cache_view_n := evalContext.GgmlView_2d(cache_v, n_tokens, n_embd_v_gqa,
				// 	int64(n_ctx*2),
				// 	int64(kv_head*2),
				// )
				// v_cache_view_n.SetName(fmt.Sprintf("v_cache_view-%d", l))
				// layer.v_cache_view_n = v_cache_view_n

				// // fmt.Println("v_cache_view_n", v_cache_view_n.DebugString())

				// // ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
				// layer.v_cache_copy = evalContext.GgmlCopy(vcur_n_transposed, v_cache_view_n)

				// // struct ggml_tensor * v =
				// // ggml_view_3d(ctx, kv.v_l[il],
				// //         n_kv, n_embd_head_v, n_head_kv,
				// //         ggml_element_size(kv.v_l[il])*n_ctx,
				// //         ggml_element_size(kv.v_l[il])*n_ctx*n_embd_head_v,
				// //         0);

				// v_n := evalContext.GgmlView_3d(cache_v, n_kv, n_embd_head_v, n_head_kv,
				// 	int64(2*n_ctx),
				// 	int64(2*n_ctx*n_embd_head_v),
				// 	0,
				// )
				// // fmt.Println("v_n", v_n.DebugString())
				// v_n.SetName(fmt.Sprintf("v-%d", l))
				// layer.v = v_n

				// v_n := evalContext.GgmlPermute(vcur_n_reshaped, 1, 2, 0, 3)
				// fmt.Println("v_n", v_n.DebugString())

				// // Final shape: n_tokens x 128 x 8
				// layer.v = v_n
				fmt.Println("layer.v", layer.v.DebugString())
			}
			// fmt.Println("n_embd_head_k", n_embd_head_k)
			// fmt.Println("n_kv", n_kv)
			// fmt.Println("n_head_kv", n_head_kv)
			// fmt.Println("kcur_n", kcur_n.DebugString())

			{
				fmt.Println("k", layer.k.DebugString())
				fmt.Println("q", layer.q.DebugString())
				fmt.Println("v", layer.v.DebugString())
				k_t := layer.k // evalContext.GgmlTranspose(layer.k)
				// k_t := evalContext.GgmlTranspose(layer.k)
				// k_t.SetName(fmt.Sprintf("k_t-%d", l))
				q_t := layer.q // evalContext.GgmlTranspose(layer.q)
				// q_t.SetName(fmt.Sprintf("q_t-%d", l))

				// A: k columns, n rows => [ne03, ne02, n, k]
				// B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
				// result is n columns, m rows => [ne03 * x, ne02 * y, m, n]

				// matmul K and Q(transposed)
				// input: K is (128 x n_tokens x 8)
				// input: Q is (128 x n_tokens x 32)
				// output: KQ is (n_tokens x n_tokens x 32)
				// The multiplication broadcasts the 8 heads of K to match the 32 heads of Q
				// The matmul also automatically transposes the second tensor
				// K Q represents the attention weights of each token in the sequence to every other token in the sequence, for each head
				fmt.Println("mulmat", k_t.DebugString(), q_t.DebugString())
				kq_n := evalContext.GgmlMulMat(layer.k, layer.q)
				// 		        // note: this op tends to require high floating point range
				// //       while for some models F16 is enough, for others it is not, so we default to F32 here
				// ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
				kq_n.SetMulMatPrecision(llamacpp.GGML_PREC_F32)
				kq_n.SetName(fmt.Sprintf("kq-%d", l))
				fmt.Println("kq_n", kq_n.DebugString())
				layer.kq = kq_n

				// fmt.Println("kq_scale", kq_scale)
				// fmt.Println("f_max_alibi_bias", f_max_alibi_bias)
				// fmt.Println("KQ_mask", KQ_mask.DebugString())

				// Mask the attention weights to prevent attending to future tokens
				// (I think this is a big part of KV cache reuse)
				// Input & output are both: (n_tokens x n_tokens x 32)
				fmt.Println("kq_n", kq_n.DebugString())
				fmt.Println("KQ_mask", KQ_mask.DebugString())
				kq_soft_max_ext_n := evalContext.GgmlSoftMax(kq_n, KQ_mask, kq_scale, f_max_alibi_bias)
				fmt.Println("kq_soft_max_ext_n", kq_soft_max_ext_n.DebugString())
				kq_soft_max_ext_n.SetName(fmt.Sprintf("kq_soft_max_ext-%d", l))
				layer.kq_soft_max_ext = kq_soft_max_ext_n

				// Multiply V and our masked attention weights
				// matmul softmax(KQ) and V
				// input: V is (n_tokens x 128 x 8)
				// input: softmax(KQ) is (n_tokens x n_tokens x 32)
				// output: KQ is (128 x n_tokens x 32)
				// The multiplication broadcasts the 8 heads of K to match the 32 heads of Q
				// The matmul also automatically transposes the second tensor
				// K Q represents the attention weights of each token in the sequence to every other token in the sequence, for each head
				fmt.Println("mulmat", layer.v.GetName(), layer.v.Shape(), kq_soft_max_ext_n.GetName(), kq_soft_max_ext_n.Shape())
				// fmt.Println("transpose", evalContext.GgmlTranspose(layer.v).DebugString())
				// kqv_n := evalContext.GgmlMulMat(kq_soft_max_ext_n, layer.v)
				kqv_n := evalContext.GgmlMulMat(layer.v, kq_soft_max_ext_n)
				kqv_n.SetName(fmt.Sprintf("kqv-%d", l))
				layer.kqv = kqv_n
				fmt.Println("kqv_n", kqv_n.DebugString())
				cur := kqv_n

				// kqv_n = evalContext.GgmlTranspose(kqv_n)
				// mulmat Tensor: Vcur-0 (reshaped) (permuted) = (f32)    PERMUTE(Vcur-0 (reshaped){128, 8, 31, 1}, ) = {31, 128, 8, 1} Tensor: kq_soft_max_ext-0 = (f32)   SOFT_MAX(kq-0{31, 31, 32, 1}, KQ_mask{31, 31, 1, 1}) = {31, 31, 32, 1}
				// kqv_n Tensor: kqv-0 = (f32)    MUL_MAT(Vcur-0 (reshaped) (permuted){31, 128, 8, 1}, kq_soft_max_ext-0{31, 31, 32, 1}) = {128, 31, 32, 1}

				// kqv_merged_n := evalContext.GgmlPermute(kqv_n, 0, 2, 1, 3)
				// // fmt.Println("kqv_merged_n", kqv_merged_n.DebugString())
				// kqv_merged_n.SetName(fmt.Sprintf("kqv_merged-%d", l))
				// layer.kqv_merged_n = kqv_merged_n
				// fmt.Println("kqv_merged_n", kqv_merged_n.DebugString())

				cur = evalContext.GgmlPermute(cur, 0, 2, 1, 3)
				fmt.Println("kqv_n GgmlPermute", cur.DebugString())
				cur = evalContext.GgmlCont_2d(cur, n_embd_head_v*n_head, n_tokens)
				fmt.Println("kqv_n GgmlCont_2d", cur.DebugString())
				// cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v*n_head, n_tokens);
				// cb(cur, "kqv_merged_cont", il);

				// fmt.Println("cont_2d", kqv_merged_n.DebugString(), n_embd_head_v*n_head, n_tokens)
				// kqv_merged_cont_n := evalContext.GgmlCont_2d(kqv_merged_n, n_embd_head_v*n_head, n_tokens)
				// kqv_merged_cont_n.SetName(fmt.Sprintf("kqv_merged_cont-%d", l))
				// layer.kqv_merged_cont_n = kqv_merged_cont_n
				// fmt.Println("kqv_merged_cont_n", kqv_merged_cont_n.DebugString())

				// blk_n_attn_output_weight_f32 := layer.attn_output_weight.Cast(evalContext, llamacpp.GGML_TYPE_F32)
				// layer.attn_output_weight = blk_n_attn_output_weight_f32

				// matmul attn_output_weight and kqv
				// input: attn_output_weight is (4096 x 4096)
				// input: kqv is (4096 x n_tokens)
				// output: kqv_out is (n_tokens x 4096)
				fmt.Println("mulmat", layer.attn_output_weight.DebugString(), cur.DebugString())
				kqv_out_n := evalContext.GgmlMulMat(layer.attn_output_weight, cur)
				kqv_out_n.SetName(fmt.Sprintf("kqv_out-%d", l))
				// kqv_out_n.GgmlSetOutput()
				// kqv_out_n.SetMulMatPrecision(llamacpp.GGML_PREC_F32)
				layer.kqv_out_n = kqv_out_n
				fmt.Println("kqv_out_n", kqv_out_n.DebugString())
			}
		}

		// FFN
		{

			cur := layer.kqv_out_n
			input := layer_input

			if l == n_layers-1 {
				// skip computing output for unused tokens
				n_tokens = n_outputs
				cur = evalContext.GetRows(cur, inp_out_ids)
				input = evalContext.GetRows(input, inp_out_ids)
			}

			ffn_inp_n := evalContext.GgmlAdd(cur, input)
			// fmt.Println("ffn_inp_n", ffn_inp_n.DebugString())
			ffn_inp_n.SetName(fmt.Sprintf("ffn_inp-%d", l))
			layer.ffn_inp = ffn_inp_n

			norm_n := evalContext.GgmlRMSNorm(ffn_inp_n, f_norm_rms_eps)
			norm_n.SetName(fmt.Sprintf("norm-%d", l))
			layer.norm = norm_n

			ffn_norm_n := evalContext.GgmlMul(norm_n, blk_n_ffn_norm_weight)
			ffn_norm_n.SetName(fmt.Sprintf("ffn_norm-%d", l))
			// fmt.Println("ffn_norm_n", ffn_norm_n.DebugString())

			// fmt.Println("blk_n_ffn_gate_weight", blk_n_ffn_gate_weight.DebugString())
			ffn_gate_n := evalContext.GgmlMulMat(blk_n_ffn_gate_weight, ffn_norm_n)
			ffn_gate_n.SetName(fmt.Sprintf("ffn_gate-%d", l))
			// fmt.Println("ffn_gate_n", ffn_gate_n.DebugString())

			ffn_silu_n := evalContext.GgmlSilu(ffn_gate_n)
			// fmt.Println("ffn_silu_n", ffn_silu_n.DebugString())
			ffn_silu_n.SetName(fmt.Sprintf("ffn_silu-%d", l))

			ffn_up_n := evalContext.GgmlMulMat(blk_n_ffn_up_weight, ffn_norm_n)
			ffn_up_n.SetName(fmt.Sprintf("ffn_up-%d", l))
			// fmt.Println("ffn_up_n", ffn_up_n.DebugString())

			ffn_gate_par_n := evalContext.GgmlMul(ffn_silu_n, ffn_up_n)
			ffn_gate_par_n.SetName(fmt.Sprintf("ffn_gate_par-%d", l))
			// fmt.Println("ffn_gate_par_n", ffn_gate_par_n.DebugString())

			ffn_out_n := evalContext.GgmlMulMat(blk_n_ffn_down_weight, ffn_gate_par_n)
			// fmt.Println("ffn_out_n", ffn_out_n.DebugString())
			ffn_out_n.SetName(fmt.Sprintf("ffn_out-%d", l))
			layer.ffn_out = ffn_out_n

			l_out_n := evalContext.GgmlAdd(ffn_out_n, ffn_inp_n)
			l_out_n.SetName(fmt.Sprintf("l_out-%d", l))
			layer.layer_output = l_out_n

			layer_input = l_out_n
		}
	}

	// Logits
	var result_output *llamacpp.GgmlTensor
	{
		last_layer := layers[n_layers-1]
		last_output := last_layer.layer_output
		// fmt.Printf("LAST OUTPUT %v\n", last_output.DebugString())
		norm := evalContext.GgmlRMSNorm(last_output, f_norm_rms_eps)
		norm.SetName("norm")
		// fmt.Println("norm", norm.DebugString())

		output_norm_weight := model.tensors["output_norm.weight"]
		if output_norm_weight == nil {
			return nil, fmt.Errorf("failed to get output.weight tensor")
		}
		result_norm := evalContext.GgmlMul(norm, output_norm_weight)
		result_norm.SetName("result_norm")
		// fmt.Println("result_norm", result_norm.DebugString())

		output_weight := model.tensors["output.weight"]
		if output_weight == nil {
			return nil, fmt.Errorf("failed to get output.weight tensor")
		}
		// Calculate logits from last layer's output.
		result_output = evalContext.GgmlMulMat(output_weight, result_norm)
		result_output.SetName("result_output")
		//	fmt.Println("result_output", result_output.DebugString())
	}

	graph, err := evalContext.NewGgmlCGraph()
	if err != nil {
		return nil, fmt.Errorf("failed to create graph: %w", err)
	}
	defer graph.Free()

	// t2 := layers[0].attn_output_weight.Cast(evalContext, llamacpp.GGML_TYPE_F32)
	// graph.BuildForwardExpand(t2)

	// Note: order matters when calling BuildForwardExpand!
	for _, layer := range layers {
		// graph.BuildForwardExpand(layer.k_cache_copy)
		// graph.BuildForwardExpand(layer.v_cache_copy)
		// graph.BuildForwardExpand(layer.kqv_merged_cont_n)
		// graph.BuildForwardExpand(layer.attn_output_weight)
		graph.BuildForwardExpand(layer.kqv_out_n)
	}
	graph.BuildForwardExpand(result_output)

	// cpu, err := ggml.GgmlBackendCpuInit()
	// if err != nil {
	// 	return fmt.Errorf("failed to create CPU backend: %w", err)
	// }
	// defer cpu.Free()

	// backends := []*ggml.GgmlBackend{
	// 	cpu,
	// }
	// scheduler, err := ggml.NewBackendScheduler(backends)
	// if err != nil {
	// 	return fmt.Errorf("failed to create scheduler: %w", err)
	// }
	// defer scheduler.Free()

	// set the input variable and parameter values
	for i, token := range input {
		tokens.SetI32_1D(i, int32(token))
		inp_pos.SetI32_1D(i, int32(i))
	}
	// tokens.SetI32_1D(0, 128000) // <begin_of_text>
	// tokens.SetI32_1D(1, 15339)  // "hello"
	// inp_pos.SetI32_1D(0, 0)
	// inp_pos.SetI32_1D(1, 1)

	// only keep last output
	// inp_out_ids.SetI32_1D(0, int32(n_tokens)-1)

	n_tokens = len(input) // Note:value cleared during graph building
	inp_out_ids.SetI32_1D(0, int32(n_tokens-1))
	fmt.Printf("inp_out_ids[%v]=%v\n", 0, int32(n_tokens-1))

	// // TODO: Do we need to zero the caches?
	// for _, cache := range caches {
	// 	cache.v.GgmlSetZero()
	// 	cache.k.GgmlSetZero()

	// 	// if i == 0 {
	// 	// 	fmt.Printf("cache.v %v\n", cache.v.Dump())
	// 	// 	fmt.Printf("cache.k %v\n", cache.k.Dump())
	// 	// }
	// }

	for i := 0; i < KQ_mask.GetDim(0); i++ {
		for j := 0; j < KQ_mask.GetDim(1); j++ {
			if i <= j {
				KQ_mask.SetF32_2D(i, j, 0)
			} else {
				KQ_mask.SetF32_2D(i, j, float32(llamacpp.GGML_NEGATIVE_INFINITY))
			}
		}
	}

	graph.ComputeWithCtx(evalContext, numThreads)

	// fmt.Printf("result: ndims %d\n", result_output.GetNDims())
	// fmt.Printf("result: nrows %d\n", result_output.GetNrows())
	// fmt.Printf("result: nelements %d\n", result_output.GetNelements())
	// fmt.Printf("result: nbytes %d\n", result_output.GetNbytes())
	// fmt.Printf("result: nbytes_pad %d\n", result_output.GetNbytesPad())
	// fmt.Printf("result: tensor type %d\n", result_output.GetTensorType())

	// fmt.Println("tokens", tokens.Dump())
	// fmt.Println("KQ_mask", KQ_mask.Dump())
	// fmt.Println("inp_embed", inp_embed.Dump())
	// // fmt.Println("norm-0", layers[0].norm.Dump())
	// // fmt.Println("attn_norm-0", layers[0].attn_norm.Dump())
	// fmt.Println("qcur_pre-0", layers[0].qcur_pre.Dump())
	// // fmt.Println("qcur_reshaped-0", layers[0].qcur_reshaped.Dump())
	// // fmt.Println("qcur-0", layers[0].qcur.Dump())
	// // fmt.Println("q-0", layers[0].q.Dump())
	// // fmt.Println("v-0", layers[0].v.Dump())

	// // fmt.Println("blk_n_attn_k_weight-0", layers[0].attn_k_weight.Dump())
	// // fmt.Println("attn_norm-0", layers[0].attn_norm.Dump())
	// // fmt.Println("kcur_n_pre-0", layers[0].kcur_n_pre.Dump())
	// // fmt.Println("kcur_n-0", layers[0].kcur_n.Dump())
	// // fmt.Println("cache_k-0", caches[0].k.Dump())
	// // fmt.Println("k_cache_view_n-0", layers[0].k_cache_view_n.Dump())
	// // fmt.Println("k_cache_copy-0", layers[0].k_cache_copy.Dump())

	// fmt.Println("vcur_n_pre-0", layers[0].vcur_n_pre.Dump())
	// // fmt.Println("vcur_n-0", layers[0].vcur_n.Dump())
	// fmt.Println("cache_v-0", caches[0].v.Dump())
	// fmt.Println("v_cache_v-0", layers[0].v_cache_view_n.Dump())
	// fmt.Println("v_cache_copy-0", layers[0].v_cache_copy.Dump())

	// fmt.Println("k-0", layers[0].k.Dump())
	// fmt.Println("q-0", layers[0].q.Dump())
	// fmt.Println("v-0", layers[0].v.Dump())

	// fmt.Println("kq-0", layers[0].kq.Dump())
	// fmt.Println("kqv-0", layers[0].kqv.Dump())
	// fmt.Println("k_cache_view-0", layers[0].k_cache_view.Dump())
	// fmt.Println("kq_soft_max_ext-0", layers[0].kq_soft_max_ext.Dump())
	// fmt.Println("kqv_merged-0", layers[0].kqv_merged_n.Dump())
	// fmt.Println("kqv_merged_cont-0", layers[0].kqv_merged_cont_n.Dump())

	// fmt.Println("blk_n_attn_output_weight", layers[0].attn_output_weight.Dump())

	// // fmt.Println("attn_output_weight-0", t2.Dump())
	// fmt.Println("kqv_out_n-0", layers[0].kqv_out_n.Dump())

	// fmt.Println("result-0", layers[0].ffn_inp.Dump())
	// fmt.Println("result-0", layers[0].ffn_out.Dump())

	// fmt.Println("result-0", layers[0].layer_output.Dump())
	// fmt.Println("result", result_output.Dump())
	// // fmt.Println("norm-1", norm_1.Dump())
	// fmt.Println("norm-2", norm_2.Dump())
	// fmt.Println("norm-3", norm_3.Dump())
	// fmt.Println("norm-4", norm_4.Dump())
	// fmt.Println("norm-5", norm_5.Dump())

	// resultData := result.DataFloat32()
	// fmt.Printf("result: data %v\n", resultData)
	return result_output, nil
}

type Tokenizer struct {
	model *llamacpp.LlamaModel
}

func NewTokenizer(modelPath string) (*Tokenizer, error) {

	modelParams := llamacpp.NewLlamaModelParams()
	model, err := llamacpp.NewLlamaModel(modelPath, modelParams)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}
	return &Tokenizer{model: model}, nil
}

func (t *Tokenizer) Free() {
	t.model.Free()
}

func (t *Tokenizer) Tokenize(ctx context.Context, prompt string) ([]llamacpp.Token, error) {

	template, ok := t.model.GetChatTemplate()
	if !ok {
		return nil, fmt.Errorf("failed to get chat template")
	}
	fmt.Println(template)

	vocab, err := t.model.GetVocab()
	if err != nil {
		return nil, fmt.Errorf("failed to get vocab: %w", err)
	}
	defer vocab.Free()
	fmt.Println(vocab.NTokens())

	// llamaContextParams := llamacpp.NewLlamaContextParams()
	// llamaContext, err := llamacpp.NewLlamaContext(model, llamaContextParams)
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to create llama context: %w", err)
	// }
	// defer llamaContext.Free()

	tokens, err := vocab.TokenizePrompt(prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to tokenize prompt: %w", err)
	}

	for _, token := range tokens {
		piece, err := vocab.ConvertTokenToString(token)
		if err != nil {
			return nil, fmt.Errorf("failed to convert token to string: %w", err)
		}
		fmt.Println(token, piece)
	}

	return tokens, nil
}

func run_inference_at_high_level(ctx context.Context) error {

	llamaContextParams := llamacpp.NewLlamaContextParams()
	modelParams := llamacpp.NewLlamaModelParams()
	llamacpp.LoadAllBackends()

	model, err := llamacpp.NewLlamaModel("/home/justinsb/ai/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", modelParams)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	defer model.Free()

	template, ok := model.GetChatTemplate()
	if !ok {
		return fmt.Errorf("failed to get chat template")
	}
	fmt.Println(template)

	vocab, err := model.GetVocab()
	if err != nil {
		return fmt.Errorf("failed to get vocab: %w", err)
	}
	defer vocab.Free()
	fmt.Println(vocab.NTokens())

	llamaContext, err := llamacpp.NewLlamaContext(model, llamaContextParams)
	if err != nil {
		return fmt.Errorf("failed to create llama context: %w", err)
	}
	defer llamaContext.Free()

	prompt := `<|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024
<|eot_id|>
What is today?
`

	tokens, err := vocab.TokenizePrompt(prompt)
	if err != nil {
		return fmt.Errorf("failed to tokenize prompt: %w", err)
	}

	for _, token := range tokens {
		piece, err := vocab.ConvertTokenToString(token)
		if err != nil {
			return fmt.Errorf("failed to convert token to string: %w", err)
		}
		fmt.Printf("Token: %d, String: %s\n", token, piece)
	}

	sampler := llamacpp.NewLlamaSampler(llamacpp.LlamaSamplerOptions{
		MinP: &llamacpp.MinPOptions{
			P:       0.05,
			MinKeep: 1,
		},
		Temp: 0.7,
		Seed: 1,
	})
	defer sampler.Free()

	batch, err := llamacpp.BatchGetOne(tokens)
	if err != nil {
		return fmt.Errorf("failed to create llama batch: %w", err)
	}
	defer batch.Free()

	for {
		// check_context_size(ctx, batch);

		if err := llamaContext.Decode(batch); err != nil {
			return fmt.Errorf("failed to decode: %w", err)
		}

		// sample the next token, check is it an end of generation?
		newToken := llamaContext.Sample(sampler, -1)
		if vocab.IsEog(newToken) {
			break
		}

		piece, err := vocab.ConvertTokenToString(newToken)
		if err != nil {
			return fmt.Errorf("failed to convert token to string: %w", err)
		}

		fmt.Printf("generated: %d %s\n", newToken, piece)

		// print_word_and_concatenate_to_response(piece, response);

		// prepare the next batch with the sampled token
		batch, err = llamacpp.BatchGetOne([]llamacpp.Token{newToken})
		if err != nil {
			return fmt.Errorf("failed to create llama batch: %w", err)
		}
	}
	return nil
}
