//
// Copyright (c) 2020 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef unique_ptr_hpp_
#define unique_ptr_hpp_

#include <memory>
#include <boost/python.hpp>

namespace boost { namespace python { namespace converter {

// specialization for std::unique_ptr
template <typename T, typename ToPython>
struct as_to_python_function<std::unique_ptr<T>, ToPython>
{
  // Assertion functions used to prevent wrapping of converters
  // which take non-const reference parameters. The T* argument in
  // the first overload ensures it isn't used in case T is a
  // reference.
  template <class U>
  static void convert_function_must_take_value_or_const_reference(U(*)(std::unique_ptr<T>), int, std::unique_ptr<T>* = 0) {}
  template <class U>
  static void convert_function_must_take_value_or_const_reference(U(*)(std::unique_ptr<T> const&), long ...) {}

  static PyObject* convert(void const* x)
  {
    convert_function_must_take_value_or_const_reference(&ToPython::convert, 1L);
    return ToPython::convert(std::move(*const_cast<std::unique_ptr<T>*>(static_cast<std::unique_ptr<T> const*>(x))));
  }
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
  static PyTypeObject const * get_pytype() { return ToPython::get_pytype(); }
#endif
};

}}} // namespace boost::python::converter


namespace boost { namespace python { namespace detail {
// specialization for std::unique_ptr
template <class T>
struct install_holder<std::unique_ptr<T>> : converter::context_result_converter
{
  install_holder(PyObject* args_)
    : m_self(PyTuple_GetItem(args_, 0)) {}

  PyObject* operator()(std::unique_ptr<T> x) const
  {
    typedef objects::pointer_holder<std::unique_ptr<T>, T> holder;
    typedef objects::instance<holder> instance_t;

    void* memory = holder::allocate(this->m_self, offsetof(instance_t, storage), sizeof(holder));
    try
    {
      (new (memory) holder(std::move(x)))->install(this->m_self);
    }
    catch(...)
    {
      holder::deallocate(this->m_self, memory);
      throw;
    }
    return none();
  }      
  PyObject* m_self;
};
}}}

#endif
