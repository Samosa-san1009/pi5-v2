// plr_purest -- PuReST pupil tracking on BOTH eyes in parallel.
//
// One process, one decoder/reader thread, and one worker thread per eye.
// Every frame is handed to both eye workers at the same time; each worker
// owns its own PuReST tracker and runs on the eye's region of interest (ROI)
// of the full-resolution frame. Nothing is re-encoded or resized before
// PuReST sees it, so diameters are in source-frame pixels.
//
// Two modes share the same worker code:
//
//   offline  --input VIDEO --out-left L.csv --out-right R.csv
//            Decodes the recorded video once and analyses EVERY frame of
//            both eyes (workers block, nothing is dropped).
//
//   stream   --stream
//            Reads raw greyscale frames from stdin while a session records
//            and prints one result line per eye per frame on stdout. If an
//            eye ever falls behind, the oldest queued frame for that eye is
//            dropped and reported ("D" line) instead of delaying the camera.
//
// ROIs are normalised (x,y,w,h in 0..1 of the frame), so they work for any
// camera resolution:  --roi-left 0,0,0.5,1  --roi-right 0.5,0,0.5,1
//
// stdin frame record (little-endian), stream mode:
//   char[4] magic "PLR1" | u32 flags | u64 seq | i64 timestamp_ns
//   | u32 width | u32 height | width*height bytes (8-bit grey, row-major)
//   flags bit0 = reset both trackers before this frame (new session)
//
// stdout lines, stream mode:
//   READY <version>
//   P <eye> <seq> <ts> <valid> <conf> <outline_conf> <major> <minor> <cx> <cy> <angle> <proc_us>
//   D <eye> <seq> <ts>
//   BYE <frames>
//
// The pupil-detection algorithms are by Thiago Santini et al. and are licensed
// for non-commercial use only; see third_party/pupil-detection-methods.

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "PuReST.h"

#define PLR_PUREST_VERSION "2.0.0"

namespace {

// ── options ──────────────────────────────────────────────────────────────────

struct NormRoi {
    double x = 0, y = 0, w = 0.5, h = 1.0;
};

struct Options {
    bool stream = false;
    std::string input, out_left, out_right;
    NormRoi roi[2] = {{0.0, 0.0, 0.5, 1.0}, {0.5, 0.0, 0.5, 1.0}};
    float min_pupil_mm = -1.0f;
    float max_pupil_mm = -1.0f;
    float min_pupil_px = -1.0f;   // pixel limits win over the mm assumptions
    float max_pupil_px = -1.0f;
    long max_frames = -1;
    int progress_every = 0;
    size_t queue_depth = 6;
    long simulate_load_us = 0;   // testing only: extra time per frame per eye
};

const char *EYE_NAME[2] = {"left", "right"};

void usage(const char *argv0) {
    std::cerr
        << "plr_purest " << PLR_PUREST_VERSION << " - PuReST on both eyes in parallel\n"
        << "offline: " << argv0 << " --input VIDEO --out-left L.csv --out-right R.csv [options]\n"
        << "stream : " << argv0 << " --stream [options]   (frames on stdin, results on stdout)\n"
        << "  --roi-left x,y,w,h        normalised left-eye ROI  (default 0,0,0.5,1)\n"
        << "  --roi-right x,y,w,h       normalised right-eye ROI (default 0.5,0,0.5,1)\n"
        << "  --min-pupil-mm F          PuRe minimum pupil diameter assumption (default 2)\n"
        << "  --max-pupil-mm F          PuRe maximum pupil diameter assumption (default 8)\n"
        << "  --min-pupil-px F          smallest pupil diameter to accept, in frame pixels\n"
        << "  --max-pupil-px F          largest pupil diameter to accept, in frame pixels\n"
        << "  --max-frames N            offline: stop after N frames (testing)\n"
        << "  --progress N              offline: 'progress <frames>' on stderr every N frames\n"
        << "  --queue-depth N           stream: frames buffered per eye before dropping (default 6)\n";
}

bool parse_roi(const std::string &text, NormRoi &roi) {
    double v[4];
    char c1, c2, c3;
    std::istringstream in(text);
    if (!(in >> v[0] >> c1 >> v[1] >> c2 >> v[2] >> c3 >> v[3])) return false;
    if (c1 != ',' || c2 != ',' || c3 != ',') return false;
    if (v[0] < 0 || v[1] < 0 || v[2] <= 0 || v[3] <= 0) return false;
    if (v[0] + v[2] > 1.0001 || v[1] + v[3] > 1.0001) return false;
    roi = {v[0], v[1], v[2], v[3]};
    return true;
}

bool parse(int argc, char **argv, Options &opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--stream") opt.stream = true;
        else if (arg == "--input") opt.input = next("--input");
        else if (arg == "--out-left") opt.out_left = next("--out-left");
        else if (arg == "--out-right") opt.out_right = next("--out-right");
        else if (arg == "--roi-left" || arg == "--roi-right") {
            const std::string value = next(arg.c_str());
            if (!parse_roi(value, opt.roi[arg == "--roi-left" ? 0 : 1])) {
                std::cerr << "bad ROI for " << arg << ": " << value << " (expected x,y,w,h in 0..1)\n";
                return false;
            }
        }
        else if (arg == "--min-pupil-mm") opt.min_pupil_mm = std::strtof(next("--min-pupil-mm").c_str(), nullptr);
        else if (arg == "--max-pupil-mm") opt.max_pupil_mm = std::strtof(next("--max-pupil-mm").c_str(), nullptr);
        else if (arg == "--min-pupil-px") opt.min_pupil_px = std::strtof(next("--min-pupil-px").c_str(), nullptr);
        else if (arg == "--max-pupil-px") opt.max_pupil_px = std::strtof(next("--max-pupil-px").c_str(), nullptr);
        else if (arg == "--max-frames") opt.max_frames = std::strtol(next("--max-frames").c_str(), nullptr, 10);
        else if (arg == "--progress") opt.progress_every = std::atoi(next("--progress").c_str());
        else if (arg == "--queue-depth") opt.queue_depth = std::max(1, std::atoi(next("--queue-depth").c_str()));
        else if (arg == "--simulate-load-us") opt.simulate_load_us = std::strtol(next("--simulate-load-us").c_str(), nullptr, 10);
        else if (arg == "--version") { std::cout << PLR_PUREST_VERSION << "\n"; std::exit(0); }
        else if (arg == "--help" || arg == "-h") { usage(argv[0]); std::exit(0); }
        else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (opt.stream) return true;
    return !(opt.input.empty() || opt.out_left.empty() || opt.out_right.empty());
}

cv::Rect pixel_roi(const NormRoi &n, int cols, int rows) {
    int x = static_cast<int>(std::lround(n.x * cols));
    int y = static_cast<int>(std::lround(n.y * rows));
    int w = static_cast<int>(std::lround(n.w * cols));
    int h = static_cast<int>(std::lround(n.h * rows));
    cv::Rect r(x, y, std::max(1, w), std::max(1, h));
    return r & cv::Rect(0, 0, cols, rows);
}

// ── work items and per-eye queue ─────────────────────────────────────────────

struct Job {
    std::shared_ptr<const cv::Mat> frame;  // full greyscale frame, shared by both eyes
    uint64_t seq = 0;
    int64_t ts = 0;
    bool reset = false;
};

class EyeQueue {
public:
    explicit EyeQueue(size_t depth) : depth_(depth) {}

    // Offline: wait for room, never drop.
    void push_blocking(Job job) {
        std::unique_lock<std::mutex> lock(mu_);
        room_.wait(lock, [&] { return items_.size() < depth_; });
        items_.push_back(std::move(job));
        ready_.notify_one();
    }

    // Stream: never wait; if full, drop the oldest frame and return it.
    bool push_dropping(Job job, Job &dropped) {
        std::lock_guard<std::mutex> lock(mu_);
        bool did_drop = false;
        if (items_.size() >= depth_) {
            dropped = std::move(items_.front());
            items_.pop_front();
            did_drop = true;
            // Keep a pending reset: the session start must not be lost.
            if (dropped.reset) job.reset = true;
        }
        items_.push_back(std::move(job));
        ready_.notify_one();
        return did_drop;
    }

    bool pop(Job &job) {
        std::unique_lock<std::mutex> lock(mu_);
        ready_.wait(lock, [&] { return closed_ || !items_.empty(); });
        if (items_.empty()) return false;
        job = std::move(items_.front());
        items_.pop_front();
        room_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        ready_.notify_all();
        room_.notify_all();
    }

private:
    size_t depth_;
    std::mutex mu_;
    std::condition_variable ready_, room_;
    std::deque<Job> items_;
    bool closed_ = false;
};

// ── one eye's tracker ────────────────────────────────────────────────────────

struct Result {
    int eye = 0;
    uint64_t seq = 0;
    int64_t ts = 0;
    bool valid = false;
    float confidence = 0, outline_confidence = 0;
    float major = NAN, minor = NAN, cx = NAN, cy = NAN, angle = NAN;
    long proc_us = 0;
};

class EyeWorker {
public:
    EyeWorker(int eye, const Options &opt, std::function<void(const Result &)> sink)
        : eye_(eye), roi_(opt.roi[eye]), queue_(opt.queue_depth), sink_(std::move(sink)),
          load_us_(opt.simulate_load_us), min_px_(opt.min_pupil_px), max_px_(opt.max_pupil_px) {
        if (opt.min_pupil_mm > 0) tracker_.minPupilDiameterMM = opt.min_pupil_mm;
        if (opt.max_pupil_mm > 0) tracker_.maxPupilDiameterMM = opt.max_pupil_mm;
    }

    // PuRe derives its pixel limits from the image diagonal d:
    //   max_px = d * maxMM / canthi_mm,   min_px = (2/3) d * minMM / canthi_mm
    // (scale-invariant, also inside PuReST tracking). Invert that so pixel
    // limits in frame pixels can be set for the current ROI size.
    void apply_pixel_limits(const cv::Size &roi) {
        if (roi == limits_for_ || (min_px_ <= 0 && max_px_ <= 0)) return;
        limits_for_ = roi;
        const float d = std::sqrt(static_cast<float>(roi.width) * roi.width + static_cast<float>(roi.height) * roi.height);
        const float canthi = tracker_.meanCanthiDistanceMM;
        if (max_px_ > 0) tracker_.maxPupilDiameterMM = max_px_ * canthi / d;
        if (min_px_ > 0) tracker_.minPupilDiameterMM = min_px_ * canthi * 1.5f / d;
    }

    void start() { thread_ = std::thread([this] { loop(); }); }
    void join() { queue_.close(); if (thread_.joinable()) thread_.join(); }
    EyeQueue &queue() { return queue_; }
    std::atomic<long> processed{0};

private:
    void loop() {
        Job job;
        while (queue_.pop(job)) {
            Result r;
            r.eye = eye_;
            r.seq = job.seq;
            r.ts = job.ts;
            const auto t0 = std::chrono::steady_clock::now();
            if (job.reset) tracker_.reset();
            if (load_us_ > 0) std::this_thread::sleep_for(std::chrono::microseconds(load_us_));
            const cv::Mat &frame = *job.frame;
            const cv::Rect roi = pixel_roi(roi_, frame.cols, frame.rows);
            if (roi.width >= 16 && roi.height >= 16) {
                apply_pixel_limits(roi.size());
                const cv::Mat view = frame(roi);  // no copy, no resize
                Pupil pupil;
                tracker_.run(view, pupil);
                const bool outline = pupil.size.width > 0 && pupil.size.height > 0 &&
                                     pupil.center.x > 0 && pupil.center.y > 0;
                r.confidence = pupil.confidence;
                r.outline_confidence = pupil.outline_confidence;
                r.valid = outline && pupil.confidence > 0.0f;
                if (r.valid && max_px_ > 0 && std::max(pupil.size.width, pupil.size.height) > max_px_ * 1.15f)
                    r.valid = false;   // outline far larger than any pupil (e.g. the iris)
                if (outline) {
                    r.major = std::max(pupil.size.width, pupil.size.height);
                    r.minor = std::min(pupil.size.width, pupil.size.height);
                    r.cx = pupil.center.x + roi.x;   // full-frame coordinates
                    r.cy = pupil.center.y + roi.y;
                    r.angle = pupil.angle;
                }
            }
            r.proc_us = static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count());
            processed.fetch_add(1);
            sink_(r);
        }
    }

    int eye_;
    NormRoi roi_;
    PuReST tracker_;
    EyeQueue queue_;
    std::function<void(const Result &)> sink_;
    long load_us_;
    float min_px_, max_px_;
    cv::Size limits_for_{0, 0};
    std::thread thread_;
};

std::string fmt(float v) {
    if (!std::isfinite(v)) return "nan";
    std::ostringstream s;
    s << std::fixed << std::setprecision(4) << v;
    return s.str();
}

// ── offline mode ─────────────────────────────────────────────────────────────

int run_offline(const Options &opt) {
    cv::VideoCapture cap(opt.input);
    if (!cap.isOpened()) {
        std::cerr << "cannot open video: " << opt.input << "\n";
        return 3;
    }
    const long total = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cerr << "total " << (total > 0 ? total : 0) << std::endl;

    const std::string paths[2] = {opt.out_left, opt.out_right};
    std::ofstream out[2];
    for (int e = 0; e < 2; ++e) {
        out[e].open(paths[e] + ".partial");
        if (!out[e]) {
            std::cerr << "cannot write: " << paths[e] << ".partial\n";
            return 4;
        }
        out[e] << "Frame,Valid,Confidence,Outline_confidence,Diameter_px,Minor_px,"
                  "Center_x,Center_y,Angle_deg,Proc_us\n";
    }
    // Each eye writes only its own file, from its own thread: no lock needed.
    auto sink = [&](const Result &r) {
        std::ofstream &o = out[r.eye];
        o << r.seq << ',' << (r.valid ? 1 : 0) << ',' << fmt(r.confidence) << ','
          << fmt(r.outline_confidence) << ',';
        if (std::isfinite(r.major))
            o << fmt(r.major) << ',' << fmt(r.minor) << ',' << fmt(r.cx) << ',' << fmt(r.cy) << ',' << fmt(r.angle);
        else
            o << ",,,,";
        o << ',' << r.proc_us << '\n';
    };
    Options offline = opt;
    offline.queue_depth = 8;
    EyeWorker left(0, offline, sink), right(1, offline, sink);
    left.start();
    right.start();

    const auto started = std::chrono::steady_clock::now();
    cv::Mat frame;
    uint64_t index = 0;
    while (cap.read(frame)) {
        if (opt.max_frames >= 0 && static_cast<long>(index) >= opt.max_frames) break;
        auto grey = std::make_shared<cv::Mat>();
        if (frame.channels() == 3) cv::cvtColor(frame, *grey, cv::COLOR_BGR2GRAY);
        else if (frame.channels() == 4) cv::cvtColor(frame, *grey, cv::COLOR_BGRA2GRAY);
        else *grey = frame.clone();
        Job job;
        job.frame = grey;
        job.seq = index;
        job.ts = 0;
        job.reset = index == 0;
        left.queue().push_blocking(job);
        right.queue().push_blocking(job);
        ++index;
        if (opt.progress_every > 0 && index % opt.progress_every == 0) {
            const long done = std::min(left.processed.load(), right.processed.load());
            std::cerr << "progress " << done << std::endl;
        }
    }
    left.join();
    right.join();

    for (int e = 0; e < 2; ++e) {
        out[e].close();
        if (!out[e]) {
            std::cerr << "error while writing " << paths[e] << "\n";
            return 5;
        }
    }
    if (index == 0) {
        std::cerr << "no frames decoded from " << opt.input << "\n";
        for (const auto &p : paths) std::remove((p + ".partial").c_str());
        return 6;
    }
    for (const auto &p : paths) {
        if (std::rename((p + ".partial").c_str(), p.c_str()) != 0) {
            std::cerr << "cannot rename " << p << ".partial\n";
            return 7;
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cerr << "done frames=" << index << " seconds=" << seconds
              << " fps=" << (seconds > 0 ? index / seconds : 0.0) << std::endl;
    return 0;
}

// ── stream mode ──────────────────────────────────────────────────────────────

bool read_exact(void *dst, size_t n) {
    auto *p = static_cast<char *>(dst);
    while (n > 0) {
        const size_t got = std::fread(p, 1, n, stdin);
        if (got == 0) return false;
        p += got;
        n -= got;
    }
    return true;
}

int run_stream(const Options &opt) {
    static char stdin_buffer[1 << 20];
    std::setvbuf(stdin, stdin_buffer, _IOFBF, sizeof(stdin_buffer));
    std::mutex out_mu;
    auto emit = [&](const std::string &line) {
        std::lock_guard<std::mutex> lock(out_mu);
        std::fwrite(line.data(), 1, line.size(), stdout);
        std::fflush(stdout);
    };
    auto sink = [&](const Result &r) {
        std::ostringstream s;
        s << "P " << EYE_NAME[r.eye] << ' ' << r.seq << ' ' << r.ts << ' ' << (r.valid ? 1 : 0) << ' '
          << fmt(r.confidence) << ' ' << fmt(r.outline_confidence) << ' ' << fmt(r.major) << ' '
          << fmt(r.minor) << ' ' << fmt(r.cx) << ' ' << fmt(r.cy) << ' ' << fmt(r.angle) << ' '
          << r.proc_us << '\n';
        emit(s.str());
    };
    EyeWorker left(0, opt, sink), right(1, opt, sink);
    EyeWorker *workers[2] = {&left, &right};
    left.start();
    right.start();
    emit(std::string("READY ") + PLR_PUREST_VERSION + "\n");

    uint64_t frames = 0;
    while (true) {
        char magic[4];
        uint32_t flags = 0, width = 0, height = 0;
        uint64_t seq = 0;
        int64_t ts = 0;
        if (!read_exact(magic, 4)) break;
        if (std::memcmp(magic, "PLR1", 4) != 0) {
            std::cerr << "bad frame header (stream out of sync)\n";
            break;
        }
        if (!read_exact(&flags, 4) || !read_exact(&seq, 8) || !read_exact(&ts, 8) ||
            !read_exact(&width, 4) || !read_exact(&height, 4)) break;
        if (width == 0 || height == 0 || width > 8192 || height > 8192) {
            std::cerr << "bad frame size " << width << "x" << height << "\n";
            break;
        }
        auto grey = std::make_shared<cv::Mat>(static_cast<int>(height), static_cast<int>(width), CV_8UC1);
        if (!read_exact(grey->data, static_cast<size_t>(width) * height)) break;
        Job job;
        job.frame = grey;
        job.seq = seq;
        job.ts = ts;
        job.reset = (flags & 1u) != 0;
        for (int e = 0; e < 2; ++e) {
            Job dropped;
            if (workers[e]->queue().push_dropping(job, dropped)) {
                std::ostringstream s;
                s << "D " << EYE_NAME[e] << ' ' << dropped.seq << ' ' << dropped.ts << '\n';
                emit(s.str());
            }
        }
        ++frames;
    }
    left.join();   // drains whatever is still queued
    right.join();
    emit("BYE " + std::to_string(frames) + "\n");
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    Options opt;
    if (!parse(argc, argv, opt)) {
        usage(argv[0]);
        return 2;
    }
    // Each eye already has its own thread; stop OpenCV from also spreading
    // every call over all cores, which would make the two eyes fight.
    cv::setNumThreads(1);
    return opt.stream ? run_stream(opt) : run_offline(opt);
}
