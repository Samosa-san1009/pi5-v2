#include "soft_flow.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/features.hpp>
#endif
#include <algorithm>
#include <cmath>
#include <vector>

namespace plr {
namespace {
float major(const Pupil &p) { return std::max(p.size.width, p.size.height); }
float minor(const Pupil &p) { return std::min(p.size.width, p.size.height); }
float median(std::vector<float> v) {
    auto mid = v.begin() + v.size() / 2;
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}
}

SoftFlow::SoftFlow(SoftFlowConfig config) : config_(config) {}

void SoftFlow::reset() {
    tracker_.reset();
    previous_.release();
    anchor_.clear();
    pending_.clear();
    lost_ = 0;
    have_frame_ = false;
    anchor_ts_ = 0;
}

void SoftFlow::limits(const cv::Size &size, float scale) {
    const float d = std::hypot(static_cast<float>(size.width), static_cast<float>(size.height));
    tracker_.minPupilDiameterMM = config_.min_mm > 0 ? config_.min_mm : 2.f;
    tracker_.maxPupilDiameterMM = config_.max_mm > 0 ? config_.max_mm : 8.f;
    if (config_.min_px > 0)
        tracker_.minPupilDiameterMM = config_.min_px * scale * tracker_.meanCanthiDistanceMM * 1.5f / d;
    if (config_.max_px > 0)
        tracker_.maxPupilDiameterMM = config_.max_px * scale * tracker_.meanCanthiDistanceMM / d;
}

bool SoftFlow::geometry(const Pupil &p, const cv::Size &size, float scale, float aspect) const {
    const float a = major(p), b = minor(p);
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(p.angle) ||
        !std::isfinite(p.center.x) || !std::isfinite(p.center.y) || b < 4 || b / a < aspect)
        return false;
    // Require the fitted ellipse to be wholly in the eye crop.
    const cv::Rect box = p.boundingRect();
    if (box.x < 1 || box.y < 1 || box.br().x >= size.width || box.br().y >= size.height)
        return false;
    const float diagonal = std::hypot(static_cast<float>(size.width), static_cast<float>(size.height));
    const float lo = config_.min_px > 0 ? config_.min_px * scale :
        (config_.min_mm > 0 ? (2.f / 3) * diagonal * config_.min_mm / tracker_.meanCanthiDistanceMM : 6.f);
    const float hi = config_.max_px > 0 ? config_.max_px * scale :
        (config_.max_mm > 0 ? diagonal * config_.max_mm / tracker_.meanCanthiDistanceMM :
                             .65f * std::min(size.width, size.height));
    return a >= lo && a <= hi;
}

float SoftFlow::contrast(const cv::Mat &gray, const Pupil &p) const {
    if (!p.hasOutline()) return 0;
    // Work only around the ellipse; no full-frame masks for each candidate.
    const int radius = static_cast<int>(std::ceil(.85f * major(p))) + 2;
    const cv::Rect box = cv::Rect(cvRound(p.center.x) - radius, cvRound(p.center.y) - radius,
                                  2 * radius + 1, 2 * radius + 1) & cv::Rect(0, 0, gray.cols, gray.rows);
    if (box.empty()) return 0;
    cv::Mat inner = cv::Mat::zeros(box.size(), CV_8U), ring = inner.clone(), near = inner.clone();
    cv::RotatedRect ellipse = p;
    ellipse.center -= cv::Point2f(static_cast<float>(box.x), static_cast<float>(box.y));
    ellipse.size *= .75f;
    cv::ellipse(inner, ellipse, cv::Scalar(255), -1);
    ellipse.size = p.size * 1.55f;
    cv::ellipse(ring, ellipse, cv::Scalar(255), -1);
    ellipse.size = p.size * 1.10f;
    cv::ellipse(near, ellipse, cv::Scalar(255), -1);
    ring.setTo(0, near);
    if (cv::countNonZero(inner) < 8 || cv::countNonZero(ring) < 8) return 0;
    cv::Scalar inner_mean, inner_std;
    cv::meanStdDev(gray(box), inner_mean, inner_std, inner);
    const float c = static_cast<float>(cv::mean(gray(box), ring)[0] - inner_mean[0]);
    // An iris-sized fit can enclose both dark pupil and bright iris. Reject
    // that mixed interior instead of accepting its superficially dark center.
    if (inner_std[0] > std::max(10.f, .45f * c)) return 0;
    return c;
}

bool SoftFlow::consistent(const Pupil &a, const Pupil &b) const {
    if (!a.hasOutline() || !b.hasOutline()) return false;
    const float ratio = major(a) / major(b);
    return cv::norm(a.center - b.center) <= std::max(6.f, .35f * major(b)) &&
           ratio >= .85f && ratio <= 1.15f;
}

Pupil SoftFlow::candidate(const cv::Mat &gray, const Pupil *prior, float scale) const {
    cv::Rect region(0, 0, gray.cols, gray.rows);
    if (prior) {
        const int radius = cvRound(1.6f * major(*prior));
        region &= cv::Rect(cvRound(prior->center.x) - radius, cvRound(prior->center.y) - radius,
                           2 * radius + 1, 2 * radius + 1);
    }
    if (region.width < 16 || region.height < 16) return {};
    cv::Mat blur;
    cv::GaussianBlur(gray(region), blur, {3, 3}, .8);
    // Multi-threshold darkness likelihood, not a calibrated neural probability.
    // The 1st and 70th percentile resist a few glints/dead pixels while
    // retaining small constricted pupils in a full-eye search.
    int hist[256] = {}, low = 0, high = 255, count = 0;
    for (int y = 0; y < blur.rows; ++y)
        for (int x = 0; x < blur.cols; ++x) ++hist[blur.at<unsigned char>(y, x)];
    for (int i = 0; i < 256; ++i) {
        count += hist[i];
        if (count < .01 * blur.total()) low = i;
        if (count >= .70 * blur.total()) { high = i; break; }
    }
    if (high - low < config_.min_contrast) return {};
    Pupil best;
    float best_score = -1;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    for (float level : {.25f, .40f, .55f}) {
        cv::Mat mask;
        cv::threshold(blur, mask, low + level * (high - low), 255, cv::THRESH_BINARY_INV);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        // Bound ellipse fits even in a noisy crop; favor substantial components.
        std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point> &a,
                                                      const std::vector<cv::Point> &b) {
            return cv::contourArea(a) > cv::contourArea(b);
        });
        int fitted = 0;
        for (const auto &contour : contours) {
            const double area = cv::contourArea(contour);
            if (area < 20 || contour.size() < 5) continue;
            if (++fitted > 12) break;
            Pupil p(cv::fitEllipse(contour));
            p.center += cv::Point2f(static_cast<float>(region.x), static_cast<float>(region.y));
            if (!geometry(p, gray.size(), scale, .72f) || (prior && !consistent(p, *prior))) continue;
            const double fill = area / (CV_PI * p.size.width * p.size.height / 4);
            const float c = contrast(gray, p);
            if (fill < .78 || fill > 1.15 || c < config_.min_contrast) continue;
            const float aspect = minor(p) / major(p);
            const float score = std::min(c / 60.f, 1.f) + aspect - static_cast<float>(std::abs(1 - fill))
                - (prior ? static_cast<float>(cv::norm(p.center - prior->center)) / major(*prior) : 0.f);
            if (score <= best_score) continue;
            // Heuristic evidence score; kept below very confident PuReST results.
            p.confidence = std::min(.90f, .70f + .10f * std::min(c / 60.f, 1.f) + .10f * aspect);
            p.outline_confidence = p.confidence;
            best = p;
            best_score = score;
        }
    }
    return best;
}

bool SoftFlow::flow(const cv::Mat &gray, Pupil &prediction) const {
    if (previous_.empty() || !anchor_.hasOutline()) return false;
    // Sparse LK on a compact, fixed-coordinate patch surrounding the last pupil.
    const int radius = cvRound(std::max(24.f, 1.2f * major(anchor_)));
    const cv::Rect patch = cv::Rect(cvRound(anchor_.center.x) - radius, cvRound(anchor_.center.y) - radius,
                                    2 * radius + 1, 2 * radius + 1) & cv::Rect(0, 0, gray.cols, gray.rows);
    cv::Mat mask = cv::Mat::zeros(patch.size(), CV_8U);
    cv::RotatedRect outline = anchor_;
    outline.center -= cv::Point2f(static_cast<float>(patch.x), static_cast<float>(patch.y));
    outline.size *= 1.7f;
    cv::ellipse(mask, outline, cv::Scalar(255), -1);
    std::vector<cv::Point2f> points, next, back;
    cv::goodFeaturesToTrack(previous_(patch), points, 32, .01, 3, mask);
    if (points.size() < 5) return false;
    std::vector<unsigned char> forward_ok, back_ok;
    std::vector<float> error, back_error;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 15, .03);
    cv::calcOpticalFlowPyrLK(previous_(patch), gray(patch), points, next, forward_ok, error,
                            {15, 15}, 2, criteria);
    cv::calcOpticalFlowPyrLK(gray(patch), previous_(patch), next, back, back_ok, back_error,
                            {15, 15}, 2, criteria);
    std::vector<float> dx, dy;
    for (size_t i = 0; i < points.size(); ++i) {
        if (forward_ok[i] && back_ok[i] && error[i] < 20 &&
            std::isfinite(next[i].x) && std::isfinite(next[i].y) &&
            cv::norm(back[i] - points[i]) <= 1.5 &&
            next[i].x >= 0 && next[i].y >= 0 && next[i].x < patch.width && next[i].y < patch.height) {
            dx.push_back(next[i].x - points[i].x);
            dy.push_back(next[i].y - points[i].y);
        }
    }
    if (dx.size() < 5) return false;
    const float x = median(dx), y = median(dy);
    std::vector<float> residual;
    for (size_t i = 0; i < dx.size(); ++i) residual.push_back(std::hypot(dx[i] - x, dy[i] - y));
    if (median(residual) > 2.5f || std::hypot(x, y) > std::max(10.f, .4f * major(anchor_))) return false;
    prediction = anchor_;
    prediction.center += cv::Point2f(x, y);
    prediction.confidence = .60f;
    prediction.outline_confidence = 0;
    return true;
}

Detection SoftFlow::run(const cv::Mat &eye, uint64_t seq, int64_t timestamp_ns) {
    const bool discontinuity = have_frame_ && (seq != last_seq_ + 1 || eye.size() != source_size_ ||
        timestamp_ns <= last_ts_ || (timestamp_ns - last_ts_) > config_.max_gap_ms * 1e6);
    if (discontinuity) reset();
    source_size_ = eye.size();
    const float scale = std::min(1.f, static_cast<float>(config_.work_size) / std::max(eye.cols, eye.rows));
    cv::Mat gray;
    if (scale < 1) cv::resize(eye, gray, {}, scale, scale, cv::INTER_AREA);
    else gray = eye;
    if (config_.glint_kernel >= 3) {
        cv::Mat median, residual, glints, cleaned = gray.clone();
        cv::medianBlur(gray, median, config_.glint_kernel);
        cv::subtract(gray, median, residual);
        cv::threshold(residual, glints, 12, 255, cv::THRESH_BINARY);
        median.copyTo(cleaned, glints);
        gray = cleaned;
    }
    limits(gray.size(), scale);
    Detection result;
    Pupil baseline;
    tracker_.run(gray, baseline);
    const bool baseline_ok = baseline.confidence >= config_.confidence &&
        geometry(baseline, gray.size(), scale, .65f) && contrast(gray, baseline) >= config_.min_contrast;
    const bool recent = anchor_.hasOutline() && lost_ <= config_.flow_frames &&
        timestamp_ns - anchor_ts_ <= config_.max_gap_ms * 1e6;
    if (baseline_ok && ((recent && consistent(baseline, anchor_)) ||
        (!recent && pending_.hasOutline() && consistent(baseline, pending_)))) {
        result.pupil = baseline;
        result.valid = true;
        result.source = "purest";
        result.visibility = "visible";
    } else {
        tracker_.reset(); // rejected geometry must not become PuReST's next prior
        const Pupil *prior = recent ? &anchor_ : nullptr;
        Pupil soft;
        const bool search = prior || lost_ % config_.reacquire_every == 0 || pending_.hasOutline();
        if (search) soft = candidate(gray, prior, scale);
        const bool round = soft.hasOutline() && minor(soft) / major(soft) >= .82f;
        // Grayscale alone cannot reproduce RITnet's eye-openness classifier.
        // Visibility requires a contrast-supported round region; unknown stays missing.
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        result.visibility = soft.hasOutline() || baseline_ok ? "visible" :
            (stddev[0] < config_.min_contrast * .25f ? "absent" : "uncertain");
        if (round && soft.confidence >= config_.confidence &&
            (prior || (pending_.hasOutline() && consistent(soft, pending_)))) {
            result.pupil = soft;
            result.valid = true;
            result.source = prior ? "soft_contrast" : "reacquired";
        } else if (recent && lost_ < config_.flow_frames && soft.hasOutline()) {
            Pupil prediction;
            if (flow(gray, prediction) && geometry(prediction, gray.size(), scale, .65f) &&
                consistent(soft, prediction) && contrast(gray, prediction) >= config_.min_contrast) {
                result.pupil = prediction;
                result.estimated = true;
                result.source = "flow_estimate";
            }
        }
        // Confirm all independent reacquisitions in a second frame, including
        // high-confidence PuReST fits after blinks. Never rewrite the first frame.
        pending_ = baseline_ok ? baseline : (round ? soft : Pupil());
    }
    if (result.valid) {
        anchor_ = result.pupil;
        anchor_ts_ = timestamp_ns;
        lost_ = 0;
        pending_.clear();
    } else {
        ++lost_;
        if (result.estimated) anchor_ = result.pupil; // retain original anchor time
        else anchor_.clear(); // no flow across blink/uncertain evidence
    }
    previous_ = gray.clone();
    last_seq_ = seq;
    last_ts_ = timestamp_ns;
    have_frame_ = true;
    // Resizing rounds the actual dimensions. Use exact inverse scales.
    if (result.pupil.hasOutline() && gray.size() != eye.size()) {
        // Transform sampled boundary points to preserve rotated ellipses.
        const float sx = static_cast<float>(eye.cols) / gray.cols;
        const float sy = static_cast<float>(eye.rows) / gray.rows;
        std::vector<cv::Point2f> ellipse_points;
        ellipse_points.reserve(32);
        const float angle = result.pupil.angle * static_cast<float>(CV_PI / 180);
        for (int i = 0; i < 32; ++i) {
            const float t = i * static_cast<float>(2 * CV_PI / 32);
            const float x = .5f * result.pupil.size.width * std::cos(t);
            const float y = .5f * result.pupil.size.height * std::sin(t);
            ellipse_points.emplace_back((result.pupil.center.x + x * std::cos(angle) - y * std::sin(angle)) * sx,
                                        (result.pupil.center.y + x * std::sin(angle) + y * std::cos(angle)) * sy);
        }
        const Pupil old = result.pupil;
        result.pupil = Pupil(cv::fitEllipse(ellipse_points), old.confidence, old.outline_confidence, 0, -1, -1);
    }
    return result;
}
} // namespace plr
