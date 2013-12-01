//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

// This file is not intended to provide a full-featured HDF5 wrapper.
// Rather, it provides very simple functions that allow OVXX to store
// and retrieve views to and from HDF5 files.
//
// Warning:
// This is work in progress; support for different value-types,
// layouts, etc., is incomplete at best.

#ifndef ovxx_io_hdf5_hpp_
#define ovxx_io_hdf5_hpp_

#include <ovxx/dda.hpp>
#include <ovxx/view.hpp>
#include <stdexcept>
#include <hdf5.h>

namespace ovxx
{
namespace hdf5
{
namespace detail
{
template <dimension_type D> struct make_domain;
template <> struct make_domain<1>
{
  static Domain<1> construct(hsize_t *dims) { return Domain<1>(*dims);}
};
template <> struct make_domain<2>
{
  static Domain<2> construct(hsize_t *dims) { return Domain<2>(dims[0], dims[1]);}
};
template <> struct make_domain<3>
{
  static Domain<3> construct(hsize_t *dims) 
  { return Domain<3>(dims[0], dims[1], dims[2]);}
};
} // namespace ovxx::hdf5::detail

template <dimension_type D>
Domain<D> make_domain(hsize_t *dims)
{ return detail::make_domain<D>::construct(dims);}

template <typename T> struct traits;
template <> struct traits<float> 
{
  static hid_t type() { return H5T_NATIVE_FLOAT;}
};
template <> struct traits<double> 
{
  static hid_t type() { return H5T_NATIVE_DOUBLE;}
};
template <typename T> struct traits;
template <> struct traits<complex<float> >
{
  static hid_t type()
  {
    hid_t tid = H5Tcreate(H5T_COMPOUND, sizeof (complex<float>));
    H5Tinsert(tid, "real", 0, H5T_NATIVE_FLOAT);
    H5Tinsert(tid, "imag", sizeof(float), H5T_NATIVE_FLOAT);
    return tid;
  }
};
template <> struct traits<complex<double> >
{
  static hid_t type()
  {
    hid_t tid = H5Tcreate(H5T_COMPOUND, sizeof (complex<double>));
    H5Tinsert(tid, "real", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(tid, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
    return tid;
  }
};
template <> struct traits<int> 
{
  static hid_t type() { return H5T_NATIVE_INT;}
};

class exception : public std::runtime_error
{
 public:
  exception(std::string const &error) : std::runtime_error(error) {}
};

class file;

class dataset
{
  friend class file;
public:
  ~dataset() { if (impl_) H5Dclose (impl_);}

  dimension_type query_dimensionality()
  {
    hid_t space = H5Dget_space(impl_);
    int D = H5Sget_simple_extent_ndims(space);
    H5Sclose(space);
    return D;
  }
  template <dimension_type D>
  Domain<D> query_extent()
  {
    hid_t space = H5Dget_space(impl_);
    hsize_t dims[D];
    int dimensionality = H5Sget_simple_extent_dims(space, dims, 0);
    H5Sclose(space);
    OVXX_PRECONDITION(D == dimensionality);
    return make_domain<D>(dims);
  }
  template <typename T>
  bool has_value_type()
  {
    hid_t type = H5Dget_type(impl_);
    htri_t success = H5Tequal(type, traits<T>::type());
    return success > 0;
  }
  template <template <typename, typename> class V, typename T, typename B>
  void read(V<T, B> v)
  {
    OVXX_PRECONDITION(impl_); // make sure this is an existing dataset.

    dimension_type const dim = V<T, B>::dim;
    if (query_dimensionality() != dim)
      OVXX_DO_THROW(std::runtime_error("incompatible dimensionality"));
    Domain<dim> dom = query_extent<dim>();
    for (dimension_type i = 0; i != dim; ++i)
      if (dom[i].size() != v.size(i))
	OVXX_DO_THROW(std::runtime_error("incompatible dimensions"));
    typedef Layout<dim, tuple<0,1,2>, dense> layout_type;
    ovxx::dda::Data<B, ovxx::dda::out, layout_type> data(v.block()); 
    herr_t status = H5Dread(impl_, traits<T>::type(),
			    H5S_ALL, H5S_ALL, H5P_DEFAULT, data.ptr());
  }

  template <template <typename, typename> class V, typename T, typename B>
  void write(V<T, B> v)
  {
    OVXX_PRECONDITION(!impl_); // make sure this is a new dataset.

    dimension_type const dim = V<T, B>::dim;
    Domain<dim> dom = block_domain<dim>(v.block());
    hsize_t dims[dim];
    for (dimension_type d = 0; d != dim; ++d) dims[d] = dom[d].size();
    hid_t type = traits<T>::type();
    hid_t space = H5Screate_simple(dim, dims, 0);
    impl_ = H5Dcreate(file_, name_.c_str(), type, space,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(space);

    typedef Layout<dim, tuple<0,1,2>, dense> layout_type;
    vsip::dda::Data<B, vsip::dda::in, layout_type> data(v.block()); 
    herr_t status = H5Dwrite(impl_, type, H5S_ALL , H5S_ALL, H5P_DEFAULT,
			     data.ptr());
  }

private:
  dataset(hid_t file, std::string const &name, bool create = false)
    : create_(create), impl_(0), file_(file), name_(name)
  {
    if (!create)
    {
      impl_ = H5Dopen(file, name.c_str(), H5P_DEFAULT);
      if (impl_ < 0)
	OVXX_DO_THROW(exception("No dataset '" + name + "'"));
    }
  }
  bool create_;
  hid_t impl_; // Note: HDF5 takes care of reference counting these...
  hid_t file_; // remember only for post-poned creation during write.
  std::string name_;
};

class file : ovxx::detail::noncopyable
{
public:
  file(std::string const &name, char mode)
  {
    if (mode == 'r')
      impl_ = H5Fopen(name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    else
      impl_ = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (impl_ < 0)
      OVXX_DO_THROW(exception("Unable to open file '" + name + "'"));
  }
  ~file() { H5Fclose(impl_);}
  static bool is_valid(std::string const &name)
  { return H5Fis_hdf5(name.c_str()) > 0;}
  dataset open_dataset(std::string const &name)
  { return dataset(impl_, name);}
  dataset create_dataset(std::string const &name)
  { return dataset(impl_, name, true);}
  
private:
  hid_t impl_;
};

// short-hand functions for convenience
template <template <typename, typename> class V, typename T, typename B>
void read(std::string const &filename, std::string const &dset, V<T, B> v)
{
  file f(filename, 'r');
  dataset ds = f.open_dataset(dset);
  ds.read(v);
}

template <template <typename, typename> class V, typename T, typename B>
void write(std::string const &filename, std::string const &dset, V<T, B> v)
{
  file f(filename, 'w');
  dataset ds = f.create_dataset(dset);
  ds.write(v);
}

} // namespace ovxx::hdf5
} // namespace ovxx

#endif
