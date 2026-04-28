# omnivoice.cpp

Local AI text-to-speech with voice cloning and voice design, powered
by GGML. C++17 port of OmniVoice (k2-fsa/OmniVoice). 646 languages,
24 kHz mono output, runs on CPU, CUDA, ROCm, Metal, Vulkan.

## Features

- Voice cloning from a reference WAV plus its transcript
- Voice design via attribute keywords (gender, age, pitch, style,
  volume, emotion)
- Auto voice with consistent speaker identity across long inputs
- Long-form synthesis with punctuation-aware text chunking, voice
  prompt promotion, cross-fade and pydub-strict silence removal
- Bit deterministic generation in greedy mode, seedable Philox PRNG
  for stochastic sampling
- Q8_0 quantisation of the 612 M parameter Qwen3 backbone
- Two CLI tools : `omnivoice-tts` (text -> WAV) and `omnivoice-codec`
  (WAV <-> RVQ codes)

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git
cd omnivoice.cpp
./buildcuda.sh        # NVIDIA GPU
./buildvulkan.sh      # AMD/Intel GPU (Vulkan)
./buildcpu.sh         # CPU only
./buildall.sh         # all backends, runtime DL loading
```

## Model conversion

Pre-converted GGUFs are available on Hugging Face :

  https://huggingface.co/Serveurperso/OmniVoice-GGUF

Drop them in `models/` and skip to the quick start. To convert from
the original checkpoint :

```
./checkpoints.sh      # hf download k2-fsa/OmniVoice -> checkpoints/
./convert.py          # 2 GGUFs in BF16 -> models/
./quantize.sh         # base LM Q8_0 (tokenizer stays at native dtype)
```

## Quick start

```
echo "Hello world." | ./build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --lang English -o hello.wav
```

Voice cloning :

```
./build/omnivoice-tts \
    --model models/omnivoice-base-Q8_0.gguf \
    --codec models/omnivoice-tokenizer-F32.gguf \
    --ref-wav ref.wav --ref-text ref.txt \
    --lang French -o out.wav < prompt.txt
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model, the
GGUF layout, the inference pipeline, every CLI flag, and the
validation results.

## Optimisation roadmap

Active speed-up tracks for the inference pipeline.

**MaskGIT prefix KV cache** (high impact, in progress). The decoder
re-computes attention K and V for the full sequence on every step,
including the invariant prefix (denoise + lang + instruct + text +
ref_audio_codes). Caching the prefix K/V and recomputing only the
target zone saves a fraction of the attention compute proportional
to `prefix / (prefix + target)`. Voice cloning chunk 0 with a long
reference is around 77 percent cacheable, expected gain 3 to 4x on
the MaskGIT phase.

**Persistent buffers across chunks** (low impact, easy). The graph
allocator is currently rebuilt per chunk. Sizing it once on the
worst-case chunk and reusing it cuts the per-chunk setup overhead.

## License

MIT. See [LICENSE](LICENSE).

Upstream model : OmniVoice by Xiaomi / k2-fsa, Apache 2.0.
Audio codec : Higgs Audio v2 (`bosonai/higgs-audio-v2-tokenizer`),
Apache 2.0.
