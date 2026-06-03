#include "mpc_wrapper.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <filesystem>

#if defined(__APPLE__)
  #include <mach-o/dyld.h>
#elif defined(__linux__)
  #include <unistd.h>
#endif

namespace strider_mpc {

namespace fs = std::filesystem;

static fs::path get_executable_path() {
  #if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    return fs::weakly_canonical(buf.c_str());
  #elif defined(__linux__)
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
    return fs::weakly_canonical(buf);
  #else
    return {};
  #endif
}

static void ensure_python_paths() {
  const fs::path exe = get_executable_path();
  const fs::path root = exe.parent_path().parent_path(); // build/ -> project root
  fs::path cand = root / "resources";
  const fs::path res = fs::weakly_canonical(cand);
  const std::string res_s = res.string();

  pybind11::module_ sys = pybind11::module_::import("sys");
  pybind11::list sys_path = sys.attr("path");

  sys_path.insert(0, res_s);
}

struct acados_wrapper::Impl {
  pybind11::object solver;

  Impl() {
    pybind11::gil_scoped_acquire gil;
    ensure_python_paths();
    pybind11::module_ mod = pybind11::module_::import("mpc_py.solver");
    solver = mod.attr("StriderNMPC")();
  }

  static MPCOutput from_dict(const pybind11::dict& d) {
    MPCOutput out;
    out.x_stage = d["x_stage"].cast<Eigen::Matrix<double, MPC_NX, param::N_STEPS_REQ>>();
    out.u_stage = d["u_stage"].cast<Eigen::Matrix<double, MPC_NU, param::N_STEPS_REQ>>();
    out.solve_ms = d["solve_ms"].cast<double>();
    out.state = d["state"].cast<std::uint8_t>();
    return out;
  }
};

acados_wrapper::acados_wrapper() : impl_(std::make_unique<Impl>()) {}
acados_wrapper::acados_wrapper(acados_wrapper&&) noexcept = default;
acados_wrapper& acados_wrapper::operator=(acados_wrapper&&) noexcept = default;

acados_wrapper::~acados_wrapper() {
  if (impl_) {
    pybind11::gil_scoped_acquire gil;
    impl_.reset();
  }
}

MPCOutput acados_wrapper::compute(const MPCInput& in) {
  pybind11::gil_scoped_acquire gil;

  pybind11::dict mpci;
  mpci["x_0"]   = in.x_0;
  mpci["u_0"]   = in.u_0;
  mpci["p"]     = in.p;
  mpci["use_delta"] = pybind11::bool_(in.use_delta);
  mpci["use_arm"] = pybind11::bool_(in.use_arm);
  mpci["steps_req"] = pybind11::int_(in.steps_req);
  mpci["epoch"]     = pybind11::int_(in.epoch);
  mpci["time"]      = pybind11::float_(std::chrono::duration<double>(in.t.time_since_epoch()).count());

  pybind11::object ret = impl_->solver.attr("compute_MPC")(mpci);
  return Impl::from_dict(ret.cast<pybind11::dict>());
}

} // namespace strider_mpc