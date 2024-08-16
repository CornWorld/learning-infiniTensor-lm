use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors, View};
use safetensors::Dtype::F32;

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            let tv = safetensor.tensor(name).unwrap();
            assert_eq!(tv.dtype(), F32);
            assert_eq!(tv.data().len() % 4, 0);
            let data = tv.data().chunks_exact(4).map(
                |c| f32::from_le_bytes(c.try_into().unwrap())
            ).collect();
            Tensor::new(data, &tv.shape().to_vec())
        };
        let get_tensor_from_layers = |name: &str| -> Vec<Tensor<f32>> {
            (0..config.num_hidden_layers).map(
                |i| get_tensor(format!("model.layers.{i}.{name}.weight").as_str())
            ).collect()
        };
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),

            rms_att_w: get_tensor_from_layers("input_layernorm"),
            wq: get_tensor_from_layers("self_attn.q_proj"),
            wk: get_tensor_from_layers("self_attn.k_proj"),
            wv: get_tensor_from_layers("self_attn.v_proj"),
            wo: get_tensor_from_layers("self_attn.o_proj"),

            rms_ffn_w: get_tensor_from_layers("post_attention_layernorm"),
            w_up: get_tensor_from_layers("mlp.up_proj"),
            w_gate: get_tensor_from_layers("mlp.gate_proj"),
            w_down: get_tensor_from_layers("mlp.down_proj"),

            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
