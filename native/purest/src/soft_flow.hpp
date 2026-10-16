#pragma once

// Causal, CPU-only adaptation of PuReST_SoftFlow_Standalone's gated recovery.
// Grayscale contrast evidence replaces neural probability/eye-class maps.
#include "PuReST.h"
#include <opencv2/core.hpp>
#include <cstdint>
#include <string>

namespace plr {
struct SoftFlowConfig {
    int work_size = 320;             // longest side of each eye; never upscale
    int flow_frames = 3;             // estimates only; never measured diameters
    int reacquire_every = 3;         // full-eye soft search while lost
    float max_gap_ms = 150;
    float min_contrast = 12;         // surrounding ring minus pupil, gray levels
    float confidence = .70f;
    int glint_kernel = 11;            // median replacement of bright local outliers; 0 disables
    float min_px = -1, max_px = -1;  // source eye pixels
    float min_mm = -1, max_mm = -1;
};

struct Detection {
    Pupil pupil;
    bool valid = false;
    bool estimated = false;
    std::string source = "missing", visibility = "uncertain";
};

class SoftFlow {
public:
    explicit SoftFlow(SoftFlowConfig config = {});
    Detection run(const cv::Mat &eye, uint64_t seq, int64_t timestamp_ns);
    void reset();

private:
    // Shared gates used for baseline and recovery.
    bool geometry(const Pupil &p, const cv::Size &size, float scale, float aspect) const;
    float contrast(const cv::Mat &gray, const Pupil &p) const;

    Pupil candidate(const cv::Mat &gray, const Pupil *prior, float scale) const;
    bool consistent(const Pupil &a, const Pupil &b) const;
    bool flow(const cv::Mat &gray, Pupil &prediction) const;
    void limits(const cv::Size &size, float scale);

    SoftFlowConfig config_;
    PuReST tracker_;
    cv::Mat previous_;
    cv::Size source_size_;
    Pupil anchor_, pending_;
    uint64_t last_seq_ = 0;
    int64_t last_ts_ = 0, anchor_ts_ = 0;
    int lost_ = 0;
    bool have_frame_ = false;
};
} // namespace plr
