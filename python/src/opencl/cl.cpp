//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/python/block.hpp>
#include <ovxx/opencl.hpp>
#include <boost/python/raw_function.hpp>

namespace bpl = boost::python;
namespace ocl = ovxx::opencl;

namespace
{
bpl::list platforms()
{
  bpl::list l;
  std::vector<ocl::platform> pls = ocl::platform::platforms();
  for (unsigned i = 0; i != pls.size(); ++i)
    l.append(pls[i]);
  return l;
}
// This function can be called both on ocl::program and ocl::context
template <typename T>
bpl::list devices(T &t)
{
  bpl::list l;
  std::vector<ocl::device> ds = t.devices();
  for (unsigned i = 0; i != ds.size(); ++i)
    l.append(ds[i]);
  return l;
}
std::auto_ptr<ocl::buffer> make_buffer(ocl::context const &c, size_t s)
{
  return std::auto_ptr<ocl::buffer>
    (new ocl::buffer(c, s, ocl::buffer::read_write));
}
void build(ocl::program &p, bpl::list devices)
{
  std::vector<ocl::device> ds;
  unsigned size = bpl::len(devices);
  for (unsigned i = 0; i != size; ++i)
    ds.push_back(bpl::extract<ocl::device>(devices[i]));
  p.build(ds);
}
// Note: We want to handle variable-numbers of arguments
//       of the form "kernel(queue, size, buf1, buf2, ...)"
//       Therefore we need to implement a generic "raw" function
//       that takes a tuple (for arguments) and a dict (for keywords)
//       and returns an object.
bpl::object run_kernel(bpl::tuple args, bpl::dict kw)
{
  ocl::kernel &k = bpl::extract<ocl::kernel&>(args[0]);
  ocl::command_queue &q = bpl::extract<ocl::command_queue&>(args[1]);
  size_t s = bpl::extract<size_t>(args[2]);
  switch (bpl::len(args) - 3)
  {
    // count down to avoid duplicating code
    case 4:
      k.set_arg(3, bpl::extract<ocl::buffer &>(args[6]));
    case 3:
      k.set_arg(2, bpl::extract<ocl::buffer &>(args[5]));
    case 2:
      k.set_arg(1, bpl::extract<ocl::buffer &>(args[4]));
    case 1:
      k.set_arg(0, bpl::extract<ocl::buffer &>(args[3]));
      break;
    default:
      throw std::runtime_error("too many arguments in kernel(...)");
  }
  k.exec(q, s);
  return bpl::object();
}

// mainly just for convenience (and validation), allow a simple
// means to inject and extract raw data to and from buffers
void write_data(ocl::command_queue &q, bpl::object o, ocl::buffer &b)
{
  // o has to either be a string or a list of floats
  bpl::extract<std::string> es(o);
  bpl::extract<bpl::list> el(o);
  if (es.check())
  {
    std::string str = es;
    q.write(str.data(), b, str.size());
  }
  else if (el.check())
  {
    bpl::list l = el;
    std::vector<float> vf(bpl::len(l));
    for (unsigned i = 0; i != vf.size(); ++i)
      vf[i] = bpl::extract<float>(l[i]);
    q.write(&*vf.begin(), b, vf.size());
  }
  else
    throw std::invalid_argument("unsupported value-type to command_queue.write()");
}
template <typename T>
bpl::object read_data(ocl::command_queue &q, ocl::buffer &b, size_t elements)
{
  std::vector<T> data(elements);
  q.read(b, &*data.begin(), elements);
  bpl::list l;
  for (unsigned i = 0; i != elements; ++i)
    l.append(data[i]);
  return l;
}

template <>
bpl::object read_data<char>(ocl::command_queue &q, ocl::buffer &b, size_t elements)
{
  std::vector<char> data(elements);
  q.read(b, &*data.begin(), elements);
  std::string str(&*data.begin(), data.size());
  return bpl::object(str);
}

} // namespace <anonymous>

void define_platform()
{
  bpl::class_<ocl::platform> platform("platform", bpl::no_init);
  platform.def("platforms", platforms);
  platform.staticmethod("platforms");
  platform.add_property("profile", &ocl::platform::profile);
  platform.add_property("version", &ocl::platform::version);
  platform.add_property("vendor", &ocl::platform::vendor);
  platform.add_property("name", &ocl::platform::name);
  platform.def("devices", devices<ocl::platform>);
  bpl::def("default_platform", ocl::default_platform);
}

void define_context()
{
  bpl::class_<ocl::context> context("context");
  context.def("devices", devices<ocl::context>);
  bpl::def("default_context", ocl::default_context);
}

void define_device()
{
  bpl::class_<ocl::device> device("device");
  bpl::def("default_device", ocl::default_device);
}

void define_command_queue()
{
  bpl::class_<ocl::command_queue> queue("queue");
  queue.def("write", write_data);
  queue.def("read_string", read_data<char>);
  queue.def("read_float", read_data<float>);
  bpl::def("default_queue", ocl::default_queue);
}

void define_buffer()
{
  bpl::class_<ocl::buffer> buffer("buffer");
  buffer.def("__init__", bpl::make_constructor(make_buffer));
  buffer.def("size", &ocl::buffer::size);
}

void define_program()
{
  bpl::class_<ocl::program> program("program");
  program.def("create_with_source", ocl::program::create_with_source);
  program.staticmethod("create_with_source");
  program.def("build", build);
  program.def("create_kernel", &ocl::program::create_kernel);
}

void define_kernel()
{
  void (ocl::kernel::*exec)(ocl::command_queue&,size_t) = &ocl::kernel::exec;
  bpl::class_<ocl::kernel> kernel("kernel");
  kernel.def("set_arg", &ocl::kernel::set_arg);
  kernel.def("exec", exec);
  kernel.def("__call__", bpl::raw_function(run_kernel, 3));
}

BOOST_PYTHON_MODULE(cl)
{
  define_platform();
  define_context();
  define_device();
  define_command_queue();
  define_buffer();
  define_program();
  define_kernel();
}
