//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_io_hdf5_hpp_
#define ovxx_io_hdf5_hpp_

#include <ovxx/dda.hpp>
#include <ovxx/view.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>

// This file does not intend to provide a full-featured HDF5 wrapper.
// Rather, it provides very simple functions that allow OVXX to store
// and retrieve views to and from HDF5 files.
//

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
  static H5T_class_t type_class() { return H5T_FLOAT;}
  static hid_t type() { return H5T_NATIVE_FLOAT;}
};
template <> struct traits<double> 
{
  static H5T_class_t type_class() { return H5T_FLOAT;}
  static hid_t type() { return H5T_NATIVE_DOUBLE;}
};
template <typename T> struct traits;
template <> struct traits<complex<float> >
{
  static H5T_class_t type_class() { return H5T_COMPOUND;}
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
  static H5T_class_t type_class() { return H5T_COMPOUND;}
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
  static H5T_class_t type_class() { return H5T_INTEGER;}
  static hid_t type() { return H5T_NATIVE_INT;}
};

class file;

template <typename T>
class dataset
{
  friend class file;
public:
  ~dataset() { H5Dclose (impl_);}
  void write(T const *data)
  {
    hid_t type = traits<T>::type();
    herr_t status = H5Dwrite(impl_, type, H5S_ALL , H5S_ALL, H5P_DEFAULT, data);
    // FIXME: this leaks for complex types
    // H5Tclose(type);
  }
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
  bool has_valid_value_type()
  {
    hid_t type = H5Dget_type(impl_);
    htri_t success = H5Tequal(type, traits<T>::type());
    return success > 0;
  }
  void read(T *data)
  {
    herr_t status = H5Dread(impl_, traits<T>::type(),
			    H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  }
private:
  template <dimension_type D>
  dataset(hid_t file, std::string const &name, Domain<D> const &dom)
  {
    hsize_t dims[D];
    for (dimension_type d = 0; d != D; ++d) dims[d] = dom[d].size();
    hid_t space = H5Screate_simple(D, dims, NULL);
    impl_ = H5Dcreate(file, name.c_str(), traits<T>::type(),
		      space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(space);
  }
  dataset(hid_t file, std::string const &name)
  {
    impl_ = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  }
  hid_t impl_; // Note: HDF5 takes care of reference counting these...
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
  }
  ~file() { H5Fclose(impl_);}
  static bool is_valid(std::string const &name)
  { return H5Fis_hdf5(name.c_str()) > 0;}
  template <typename T>
  dataset<T> get_dataset(std::string const &name)
  { return dataset<T>(impl_, name);}
  template <typename T, dimension_type D>
  dataset<T> create_dataset(std::string const &name, Domain<D> const &dom)
  { return dataset<T>(impl_, name, dom);}
  
private:
  hid_t impl_;
};


template <template <typename, typename> class V, typename B, typename T>
void write(std::string const &filename, V<T, B> v)
{
  dimension_type const dim = V<T, B>::dim;
  file f(filename, 'w');
  dataset<T> d = f.create_dataset<T>("data", block_domain<dim>(v.block()));
  typedef Layout<dim, tuple<0,1,2>, dense> layout_type;
  vsip::dda::Data<B, vsip::dda::in, layout_type> data(v.block()); 
  d.write(data.ptr());
}

template <template <typename, typename> class V, typename B, typename T>
void read(std::string const &filename, V<T, B> v)
{
  dimension_type const dim = V<T, B>::dim;
  if (!file::is_valid(filename))
    OVXX_DO_THROW(std::runtime_error("invalid file-format"));
  file f(filename, 'r');
  dataset<T> d = f.get_dataset<T>("data");
  if (!d.has_valid_value_type())
    OVXX_DO_THROW(std::runtime_error("invalid value-type"));
  if (d.query_dimensionality() != dim)
    OVXX_DO_THROW(std::runtime_error("incompatible dimensionality"));
  Domain<dim> dom = d.template query_extent<dim>();
  std::cout << "dimensions " << dom << std::endl;
  for (dimension_type i = 0; i != dim; ++i)
    if (dom[i].size() != v.size(i))
      OVXX_DO_THROW(std::runtime_error("incompatible dimensions"));
  // TODO: check value-type
  typedef Layout<dim, tuple<0,1,2>, dense> layout_type;
  vsip::dda::Data<B, vsip::dda::out, layout_type> data(v.block()); 
  d.read(data.ptr());
}

} // namespace ovxx::hdf5
} // namespace ovxx

#endif
