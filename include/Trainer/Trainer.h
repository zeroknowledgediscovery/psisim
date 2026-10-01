#ifndef LSM_TRAINER_H
#define LSM_TRAINER_H
#include <filesystem>
#include <string>

struct LSMTrainOptions {
    double alpha = 0.1;
    int threads = 1;
    int start = 0;
    int end = -1;
    bool skip_parse = false;
    std::string subset_mode = "auto";
    int max_exact_levels = 20;
    int fast_levels = 16;
    // Set false when the caller configures the process-wide training knobs once
    // before launching independent fits concurrently.
    bool configure_globals = true;
};

void train_lsm_model(const std::filesystem::path& csv,
                     const std::filesystem::path& outdir,
                     const LSMTrainOptions& options);
#endif
