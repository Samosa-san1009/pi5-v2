#include "eye_locator.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace plr {
EyeLocator::EyeLocator(EyeLocatorConfig config) : config_(std::move(config)) {
    if (!config_.automatic) return;
    if (!face_.load(config_.cascade_dir + "/haarcascade_frontalface_default.xml") ||
        !eyes_.load(config_.cascade_dir + "/haarcascade_eye_tree_eyeglasses.xml"))
        throw std::runtime_error("eye localization cascades missing; set --cascade-dir or rebuild/install models");
}

bool EyeLocator::plausible_pair(const cv::Rect &a, const cv::Rect &b) {
    const float width = .5f * (a.width + b.width);
    const float dx = std::abs((a.x + .5f * a.width) - (b.x + .5f * b.width));
    const float dy = std::abs((a.y + .5f * a.height) - (b.y + .5f * b.height));
    const float ratio = static_cast<float>(a.width) / std::max(1, b.width);
    return ratio >= .65f && ratio <= 1.55f && dx >= width && dx <= 4.5f * width && dy <= .45f * width;
}

bool EyeLocator::find_pair(const cv::Mat &small, const cv::Rect &region, cv::Rect (&pair)[2]) {
    if (region.width < 40 || region.height < 20) return false;
    std::vector<cv::Rect> found;
    eyes_.detectMultiScale(small(region), found, 1.1, 4, 0, {14, 14},
                          {std::max(16, small.cols / 3), std::max(16, small.cols / 3)});
    float best = -1;
    for (size_t i = 0; i < found.size(); ++i) {
        for (size_t j = i + 1; j < found.size(); ++j) {
            if (!plausible_pair(found[i], found[j])) continue;
            const float dy = std::abs(found[i].y - found[j].y);
            const float score = found[i].width + found[j].width - 2 * dy;
            if (score <= best) continue;
            best = score;
            pair[0] = found[i] + region.tl();
            pair[1] = found[j] + region.tl();
            if (pair[0].x > pair[1].x) std::swap(pair[0], pair[1]);
        }
    }
    return best >= 0;
}

EyeRegions EyeLocator::locate(const cv::Mat &gray, uint64_t seq, int64_t ts, bool reset) {
    const auto start = std::chrono::steady_clock::now();
    if (reset || !have_frame_ || size_ != gray.size() || ts <= last_ts_ ||
        ts - last_ts_ > config_.ttl_ms * 1e6) {
        cached_ = {};
        since_search_ = config_.interval;
    }
    have_frame_ = true;
    size_ = gray.size();
    (void)seq;
    last_ts_ = ts;
    const bool expired = cached_.eye[0].empty() || ts - detected_at_ > config_.ttl_ms * 1e6;
    // Decimated preview still needs a refresh before its boxes expire; retry
    // timing is based on the last search, including unsuccessful searches.
    if (++since_search_ >= config_.interval || ts - last_search_ts_ >= config_.ttl_ms * .5e6) {
        since_search_ = 0;
        last_search_ts_ = ts;
        cv::Mat small, equalized;
        const float scale = std::min(1.f, 320.f / std::max(gray.cols, gray.rows));
        cv::resize(gray, small, {}, scale, scale, cv::INTER_AREA);
        cv::equalizeHist(small, equalized);
        const cv::Rect bounds(0, 0, small.cols, small.rows);
        cv::Rect pair[2], face;
        bool found = false;
        std::string source = "tracked_eyes";
        if (!expired) {
            const cv::Rect joined = cached_.eye[0] | cached_.eye[1];
            const int margin = std::max(cached_.eye[0].height, cached_.eye[1].height);
            const cv::Rect nearby(cvRound((joined.x - .4f * margin) * scale),
                                  cvRound((joined.y - .75f * margin) * scale),
                                  cvRound((joined.width + .8f * margin) * scale),
                                  cvRound((joined.height + 1.5f * margin) * scale));
            found = find_pair(equalized, nearby & bounds, pair);
        }
        std::vector<cv::Rect> faces;
        if (!found) face_.detectMultiScale(equalized, faces, 1.05, 4, 0, {80, 80});
        if (!found && !faces.empty()) {
            face = *std::max_element(faces.begin(), faces.end(), [](const cv::Rect &a, const cv::Rect &b) {
                return a.area() < b.area();
            });
            const cv::Rect upper(face.x, face.y + cvRound(.10 * face.height), face.width,
                                 cvRound(.50 * face.height));
            found = find_pair(equalized, upper & bounds, pair);
            source = "face_eyes";
        }
        if (!found) {
            // Close rig views often crop the forehead/chin and defeat face cascades.
            // Require a matched eye pair; never send frame halves to the pupil tracker.
            found = find_pair(equalized, {0, 0, small.cols, cvRound(.68 * small.rows)}, pair);
            face = {};
            source = "eye_pair";
        }
        if (found) {
            const float sx = static_cast<float>(gray.cols) / small.cols;
            const float sy = static_cast<float>(gray.rows) / small.rows;
            EyeRegions next;
            next.source = source;
            next.face = {cvRound(face.x * sx), cvRound(face.y * sy), cvRound(face.width * sx), cvRound(face.height * sy)};
            for (int screen = 0; screen < 2; ++screen) {
                const auto &box = pair[screen];
                const float cx = (box.x + .5f * box.width) * sx;
                const float cy = (box.y + .5f * box.height) * sy;
                const int w = cvRound(.82f * box.width * sx), h = cvRound(.62f * box.height * sy);
                const int eye = config_.swap_eyes ? screen : 1 - screen;
                cv::Rect crop(cvRound(cx - w * .5f), cvRound(cy - h * .5f), w, h);
                crop &= cv::Rect(0, 0, gray.cols, gray.rows);
                const auto &old = cached_.eye[eye];
                // Suppress cascade jitter; moving the crop resets pupil coordinates.
                if (!expired && !old.empty() && cv::norm((crop.tl() + crop.br()) * .5 - (old.tl() + old.br()) * .5) <= 4 &&
                    std::abs(crop.width - old.width) <= .15f * old.width) crop = old;
                next.eye[eye] = crop;
            }
            cached_ = next;
            detected_at_ = ts;
        } else if (expired) cached_ = {};
    }
    if (!cached_.eye[0].empty() && ts - detected_at_ > config_.ttl_ms * 1e6) cached_ = {};
    EyeRegions result = cached_;
    if (!result.eye[0].empty() && ts != detected_at_) result.source += "_held";
    result.processing_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}
}
