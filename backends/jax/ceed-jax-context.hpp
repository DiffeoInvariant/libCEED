#ifndef CEED_JAX_CONTEXT_HEADER
#define CEED_JAX_CONTEXT_HEADER
#include "ceed-jax-types.hpp"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include <string>
#include <memory>
#include <function>
#include <vector>

namespace ceed {
  namespace xla {
    class Context
    {
      bool                                       m_usingCpuDevice;
      bool                                       m_usingGpuDecise;
      std::unique_ptr<xla::PjRtClient>           m_xla_client;

      
      
    public:

      Context(bool is_gpu = false);

      bool usingCpuDevice() const noexcept;

      bool usingGpuDevice() const noexcept;

      xla::PjRtClient* client() const noexcept;

      static Context* from(Ceed ceed);
    };

  } // namespace xla
} // namespace ceed

	

      

#endif // CEED_JAX_CONTEXT_HEADER
