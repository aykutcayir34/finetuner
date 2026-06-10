## GGUF · Ollama · LM Studio

🇹🇷 Bu model GGUF'a da dönüştürüldü (f16, llama.cpp `convert_hf_to_gguf.py` ile).
Kendi GGUF'unuzu üretmek için:
🇬🇧 The model also converts cleanly to GGUF (f16, via llama.cpp's
`convert_hf_to_gguf.py`). To produce your own GGUF:

```bash
# 1. Fuse adapters into the base (dequantized bf16):
mlx_lm.fuse --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --adapter-path adapters --save-path fused --dequantize
# 2. Convert with llama.cpp:
python llama.cpp/convert_hf_to_gguf.py fused --outfile model-f16.gguf --outtype f16
```

### Ollama

```bash
cat > Modelfile <<'EOF'
FROM ./model-f16.gguf
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER stop <|eot_id|>
PARAMETER temperature 0.7
PARAMETER repeat_penalty 1.15
EOF
ollama create llama32-turkish-alpaca -f Modelfile
ollama run llama32-turkish-alpaca "Sağlıklı yaşam için üç öneri ver."
```

### LM Studio

🇹🇷 LM Studio, Mac'te **MLX modellerini doğrudan** çalıştırır: uygulama içi
aramaya `{repo_id}` yazıp indirin — bu repo MLX formatındadır. Alternatif
olarak ürettiğiniz GGUF dosyasını **My Models → Import** ile yükleyebilirsiniz.
🇬🇧 LM Studio runs **MLX models natively** on Mac: search `{repo_id}` inside
the app and download — this repo is MLX-format. Alternatively import your
GGUF file via **My Models → Import**.
