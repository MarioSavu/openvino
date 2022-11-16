// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"

#include "gna/gna_config.hpp"
#include "gpu/gpu_config.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"

#include "benchmark_app.hpp"
#include "infer_request_wrap.hpp"
#include "inputs_filling.hpp"
#include "progress_bar.hpp"
#include "remote_tensors_filling.hpp"
#include "statistics_report.hpp"
#include "utils.hpp"
// clang-format on

static const size_t progressBarDefaultTotalCount = 1000;

bool parse_and_check_command_line(int argc, char* argv[]) {
    // ---------------------------Parsing and validating input
    // arguments--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help || FLAGS_h) {
        show_usage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_m.empty()) {
        show_usage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_latency_percentile > 100 || FLAGS_latency_percentile < 1) {
        show_usage();
        throw std::logic_error("The percentile value is incorrect. The applicable values range is [1, 100].");
    }
    if (FLAGS_api != "async" && FLAGS_api != "sync") {
        throw std::logic_error("Incorrect API. Please set -api option to `sync` or `async` value.");
    }
    if (!FLAGS_hint.empty() && FLAGS_hint != "throughput" && FLAGS_hint != "tput" && FLAGS_hint != "latency" &&
        FLAGS_hint != "none") {
        throw std::logic_error("Incorrect performance hint. Please set -hint option to"
                               "`throughput`(tput), `latency' value or 'none'.");
    }
    if (!FLAGS_report_type.empty() && FLAGS_report_type != noCntReport && FLAGS_report_type != averageCntReport &&
        FLAGS_report_type != detailedCntReport) {
        std::string err = "only " + std::string(noCntReport) + "/" + std::string(averageCntReport) + "/" +
                          std::string(detailedCntReport) +
                          " report types are supported (invalid -report_type option value)";
        throw std::logic_error(err);
    }

    if ((FLAGS_report_type == averageCntReport) && ((FLAGS_d.find("MULTI") != std::string::npos))) {
        throw std::logic_error("only " + std::string(detailedCntReport) + " report type is supported for MULTI device");
    }

    bool isNetworkCompiled = fileExt(FLAGS_m) == "blob";
    bool isPrecisionSet = !(FLAGS_ip.empty() && FLAGS_op.empty() && FLAGS_iop.empty());
    if (isNetworkCompiled && isPrecisionSet) {
        std::string err = std::string("Cannot set precision for a compiled network. ") +
                          std::string("Please re-compile your network with required precision "
                                      "using compile_tool");

        throw std::logic_error(err);
    }
    return true;
}

/**
 * @brief The entry point of the benchmark application
 */
int main(int argc, char* argv[]) {
    std::shared_ptr<StatisticsReport> statistics;
    try {
        ov::CompiledModel compiledModel;
        ov::Core core;

        if (!parse_and_check_command_line(argc, argv)) {
            return 0;
        }

        std::ifstream modelStream(FLAGS_m, std::ios_base::binary | std::ios_base::in);
        if (!modelStream.is_open()) {
            throw std::runtime_error("Cannot open model file " + FLAGS_m);
        }
        compiledModel = core.import_model(modelStream, "VPUX", {});
        modelStream.close();

        slog::info << "Original model I/O paramteters:" << slog::endl;
        printInputAndOutputsInfoShort(compiledModel);

    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;

        if (statistics) {
            statistics->add_parameters(StatisticsReport::Category::EXECUTION_RESULTS,
                                       {StatisticsVariant("error", "error", ex.what())});
            statistics->dump();
        }

        return 3;
    }

    return 0;
}
