// pipeline.hpp — a small, extensible image-processing pipeline.
//
// A Stage is any named transform Image -> Image. A Pipeline chains stages in
// order. New processing steps are added by implementing Stage and appending
// them; no existing code changes. This is the extension point the toolkit is
// designed around (add a stage, not edit the core).
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "wproc/image.hpp"

namespace wproc {

class Stage {
public:
    virtual ~Stage() = default;
    virtual const std::string& name() const = 0;
    virtual Image apply(const Image& in) const = 0;
};

// Adapt any callable (Image -> Image) into a Stage without a new class.
class LambdaStage final : public Stage {
public:
    LambdaStage(std::string name, std::function<Image(const Image&)> fn)
        : name_(std::move(name)), fn_(std::move(fn)) {}
    const std::string& name() const override { return name_; }
    Image apply(const Image& in) const override { return fn_(in); }

private:
    std::string name_;
    std::function<Image(const Image&)> fn_;
};

class Pipeline {
public:
    Pipeline& add(std::unique_ptr<Stage> stage) {
        stages_.push_back(std::move(stage));
        return *this;
    }
    Pipeline& add(std::string name, std::function<Image(const Image&)> fn) {
        return add(std::make_unique<LambdaStage>(std::move(name), std::move(fn)));
    }

    Image run(const Image& input) const {
        Image cur = input;
        for (const auto& s : stages_) cur = s->apply(cur);
        return cur;
    }

    std::size_t size() const { return stages_.size(); }
    const std::string& stage_name(std::size_t i) const { return stages_[i]->name(); }

private:
    std::vector<std::unique_ptr<Stage>> stages_;
};

}  // namespace wproc
