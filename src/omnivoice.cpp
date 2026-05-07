// omnivoice.cpp: public ABI implementation.
//
// Every entry declared in omnivoice.h lives here under one extern "C" block
// so the symbols carry C linkage and are linkable from C, Rust, Go, Python
// ctypes and any other binding generator. The struct OmniVoice opaque
// handle owns one BackendPair, one PipelineTTS, one PipelineCodec
// (optional), one BPETokenizer and one VoiceDesign instance. ov_init walks
// the load chain in dependency order and unwinds whatever it already
// allocated when any step fails. ov_free mirrors that order in reverse.

#include "omnivoice.h"

#include "backend.h"
#include "bpe.h"
#include "pipeline-codec.h"
#include "pipeline-tts.h"
#include "version.h"
#include "voice-design.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Internal definition of the opaque handle. C++ types are fine here
// because nothing in this struct ever crosses the public ABI boundary :
// callers only ever see `struct OmniVoice *`.
struct OmniVoice {
    BackendPair   bp;
    PipelineTTS   pt;
    PipelineCodec pc;
    BPETokenizer  tok;
    VoiceDesign   vd;
    bool          codec_loaded;
};

extern "C" {

const char * ov_version(void) {
    // Built once at first call. The C++11 magic static guarantees the
    // initialiser runs exactly once across threads.
    static const std::string s = [] {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%d.%d.%d (%s)", OV_VERSION_MAJOR, OV_VERSION_MINOR, OV_VERSION_PATCH,
                      OMNIVOICE_VERSION);
        return std::string(buf);
    }();
    return s.c_str();
}

void ov_audio_free(struct ov_audio * a) {
    if (!a) {
        return;
    }
    if (a->samples) {
        std::free(a->samples);
    }
    a->samples     = nullptr;
    a->n_samples   = 0;
    a->sample_rate = 0;
    a->channels    = 0;
}

void ov_init_default_params(struct ov_init_params * p) {
    p->model_path = nullptr;
    p->codec_path = nullptr;
    p->use_fa     = true;
    p->clamp_fp16 = false;
}

void ov_tts_default_params(struct ov_tts_params * p) {
    p->text                    = nullptr;
    p->lang                    = nullptr;
    p->instruct                = nullptr;
    p->T_override              = 0;
    p->chunk_duration_sec      = 15.0f;
    p->chunk_threshold_sec     = 30.0f;
    p->denoise                 = true;
    p->preprocess_prompt       = true;
    p->mg_num_step             = 32;
    p->mg_guidance_scale       = 2.0f;
    p->mg_t_shift              = 0.1f;
    p->mg_layer_penalty_factor = 5.0f;
    p->mg_position_temperature = 5.0f;
    p->mg_class_temperature    = 0.0f;
    p->mg_seed                 = 42;
    p->ref_audio_tokens        = nullptr;
    p->ref_T                   = 0;
    p->ref_audio_24k           = nullptr;
    p->ref_n_samples           = 0;
    p->ref_text                = nullptr;
    p->dump_dir                = nullptr;
    p->cancel                  = nullptr;
    p->cancel_user_data        = nullptr;
}

struct OmniVoice * ov_init(const struct ov_init_params * params) {
    if (!params || !params->model_path) {
        std::fprintf(stderr, "[OmniVoice] ERROR: ov_init requires a model_path\n");
        return nullptr;
    }

    std::fprintf(stderr, "[OmniVoice] omnivoice.cpp %s\n", ov_version());

    // new OmniVoice() value-initialises every field : POD aggregates
    // (BackendPair, PipelineTTS, PipelineCodec) are zero-init, std
    // containers in BPETokenizer construct empty, codec_loaded falls to
    // false. Only VoiceDesign needs explicit population below.
    OmniVoice * ov = new OmniVoice();

    voice_design_init(&ov->vd);

    // Backend init is shared (refcounted) across modules in the same
    // binary, so ov_init / ov_free pairs balance the refcount cleanly.
    ov->bp = backend_init("LM");
    if (!ov->bp.backend) {
        delete ov;
        return nullptr;
    }

    if (!pipeline_tts_load(&ov->pt, params->model_path, ov->bp, params->use_fa, params->clamp_fp16)) {
        backend_release(ov->bp.backend, ov->bp.cpu_backend);
        delete ov;
        return nullptr;
    }

    // BPE tokenizer payload lives inside the same LM GGUF as the weights.
    // Load the base vocab + the OmniVoice-specific special tokens in one
    // shot.
    if (!load_bpe_from_gguf(&ov->tok, params->model_path) ||
        !bpe_load_omnivoice_specials(&ov->tok, params->model_path)) {
        pipeline_tts_free(&ov->pt);
        backend_release(ov->bp.backend, ov->bp.cpu_backend);
        delete ov;
        return nullptr;
    }

    if (params->codec_path) {
        if (!pipeline_codec_load(&ov->pc, params->codec_path, ov->bp)) {
            pipeline_tts_free(&ov->pt);
            backend_release(ov->bp.backend, ov->bp.cpu_backend);
            delete ov;
            return nullptr;
        }
        ov->codec_loaded = true;
    }

    return ov;
}

void ov_free(struct OmniVoice * ov) {
    if (!ov) {
        return;
    }
    if (ov->codec_loaded) {
        pipeline_codec_free(&ov->pc);
    }
    pipeline_tts_free(&ov->pt);
    backend_release(ov->bp.backend, ov->bp.cpu_backend);
    delete ov;
}

enum ov_status ov_synthesize(struct OmniVoice * ov, const struct ov_tts_params * params, struct ov_audio * out) {
    if (!ov || !params || !out) {
        if (out) {
            ov_audio_free(out);
        }
        return OV_STATUS_INVALID_PARAMS;
    }
    if (!ov->codec_loaded) {
        ov_audio_free(out);
        std::fprintf(stderr, "[OmniVoice] ERROR: ov_synthesize requires a codec-loaded handle\n");
        return OV_STATUS_INVALID_PARAMS;
    }
    return pipeline_tts_synthesize(&ov->pt, &ov->pc, &ov->tok, &ov->vd, params, out);
}

int ov_duration_sec_to_tokens(const struct OmniVoice * ov, float duration_sec) {
    if (!ov || !ov->codec_loaded) {
        std::fprintf(stderr, "[OmniVoice] ERROR: ov_duration_sec_to_tokens requires a codec-loaded handle\n");
        return 1;
    }
    return pipeline_tts_duration_sec_to_tokens(&ov->pc, duration_sec);
}

}  // extern "C"
