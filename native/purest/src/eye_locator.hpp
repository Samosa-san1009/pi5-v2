#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/xobjdetect.hpp>
#endif
#include <cstdint>
#include <string>

namespace plr {
struct EyeLocatorConfig {
    bool automatic = true;
    bool swap_eyes = false;
    int interval = 6;
    float ttl_ms = 500;
    std::string cascade_dir;
};

struct EyeRegions {
    cv::Rect eye[2]; // anatomical left, right for a non-mirrored camera
    cv::Rect face;
    std::string source = "missing";
    long processing_us = 0;
};

class EyeLocator {
public:
    explicit EyeLocator(EyeLocatorConfig config);
    EyeRegions locate(const cv::Mat &gray, uint64_t seq, int64_t ts, bool reset);
    static bool plausible_pair(const cv::Rect &a, const cv::Rect &b);

private:
    bool find_pair(const cv::Mat &small, const cv::Rect &region, cv::Rect (&pair)[2]);
    EyeLocatorConfig config_;
    cv::CascadeClassifier face_, eyes_;
    EyeRegions cached_;
    cv::Size size_;
    int64_t detected_at_ = 0, last_ts_ = 0;
    int since_search_ = 0;
    bool have_frame_ = false;
};
}
