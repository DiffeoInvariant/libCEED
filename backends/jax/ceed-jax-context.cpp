#include "ceed-jax-context.hpp"

namespace ceed {
  namespace xla {

    Context::Context(bool is_gpu)
      : m_usingCpuDevice{!is_gpu},
	m_usingGpuDevice{is_gpu}
      {
	if (m_usingCpuDevice) {
	  m_xla_client = xla::GetCpuClient(/*asynchronous=*/true).ConsumeValueOrDie();
	} else {
	    m_xla_client = xla::GetGpuClient(/*asynchronous=*/true).ConsumeValueOrDie();
	}
      }

    bool Context::usingCpuDevice() const noexcept
    {
      return m_usingCpuDevice;
    }

    bool Context::usingGpuDevice() const noexcept
    {
      return m_usingGpuDevice;
    }

    xla::PjRtClient* Context::client() const noexcept
    {
      if (m_xla_client == nullptr) {
	return nullptr;
      }
      return m_xla_client.get();
    }

    Context* Context::from(Ceed ceed)
    {
      if (!ceed) {
	return nullptr;
      }
      Context *ctx;
      CeedGetData(ceed, (void**)&ctx);
      return ctx;
    }


  } // namespace xla
} // namespace ceed
