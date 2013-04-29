dnl
dnl Copyright (c) 2007 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([OVXX_CHECK_MPI],
[

# By default we will probe for MPI and use it if it exists.  If it
# does not exist, and no explicit MPI service is requested, 
# we will configure a serial VSIPL++ library.
#
# If the user specifies an MPI backend either by an explicit
# --enable-mpi[=<backend>] or by specifying the prefix to 
# MPI, then we search for MPI and issue an error if 
# it does not exist.
#
# If the user specifies that no MPI backend should be used 
# (with --disable-mpi), then we do not search for it and 
# configure a serial OpenVSIP library.

AC_ARG_ENABLE([mpi],
  AS_HELP_STRING([--enable-mpi],
                 [Use MPI. Available backends are:
                  openmpi, mpich2, intelmpi.
                  In addition, the value 'probe' causes configure to try.]),,
  [enable_mpi=probe])

AC_ARG_WITH(mpi_prefix,
  AS_HELP_STRING([--with-mpi-prefix=PATH],
                 [Specify the installation prefix of the MPI library.  Headers
                  must be in PATH/include; libraries in PATH/lib.]),
  dnl If the user specified --with-mpi-prefix, they mean to use MPI for sure.
  [
    if test -z $enable_mpi
    then enable_mpi=yes
    fi
  ])

AC_ARG_WITH(mpi_prefix64,
  AS_HELP_STRING([--with-mpi-prefix64=PATH],
                 [Specify the installation prefix of the MPI library.  Headers
                  must be in PATH/include64; libraries in PATH/lib64.]),
  dnl If the user specified --with-mpi-prefix64, they mean to use MPI for sure.
  [
    if test -z $enable_mpi
    then enable_mpi=yes
    fi
  ])

AC_ARG_WITH(mpi_cxxflags,
  AS_HELP_STRING([--with-mpi-cxxflags=FLAGS],
                 [Specify the C++ compiler flags used to compile with MPI.]))

AC_ARG_WITH(mpi_libs,
  AS_HELP_STRING([--with-mpi-libs=LIBS],
                 [Specify the linker library flags used to compile with MPI.]))

# If the user specified an MPI prefix, they definitely want MPI.
# However, we need to avoid overwriting the value of $with_mpi
# if the user set it (i.e. '--with-mpi=mpipro').

if test -n "$with_mpi_prefix" -o -n "$with_mpi_prefix64"
then
  if test "$enable_mpi" == "no"
  then AC_MSG_RESULT([MPI disabled, but MPI prefix given.])
  elif test "$enable_mpi" == "probe"
  then enable_mpi="yes"
  fi
fi

MPI_CPPFLAGS=
MPI_LDFLAGS=
MPI_LIBS=
MPI_BACKEND=none

if test "$enable_mpi" != "no"
then

  ############################################################################
  # First step: Find any required compiler flags (CPPFLAGS, LDFLAGS, LIBS).
  #             For some backends that means asking mpicc, for other 
  #             pkg-config, yet others assume default paths to be correct.
  ############################################################################
  if test -n "$with_mpi_cxxflags" -a -n "$with_mpi_libs"
  then
    MPI_CPPFLAGS="$with_mpi_cxxflags"
    MPI_LIBS="$with_mpi_libs"
    MPI_BACKEND=$enable_mpi
    
  else
    case "$enable_mpi" in
      # If the user wants one of these we look for mpicc to provide
      # compiler flags:
      openmpi | lam | mpich2 | intelmpi | probe | yes)

        if test -n "$with_mpi_prefix"
        then mpi_path="$with_mpi_prefix/bin"
        elif test -n "$with_mpi_prefix64"
        then mpi_path="$with_mpi_prefix64/bin"
        else mpi_path="$PATH"
        fi

        AC_PATH_PROGS(MPICC, mpicc hcc mpcc mpcc_r mpxlc, none, $mpi_path)
        AC_PATH_PROGS(MPICXX, mpic++ mpicxx mpiCC mpCC hcp, none, $mpi_path)
        AC_PATH_PROGS(MPI_RUN, mpirun, none, $mpi_path)

        if test ! "$MPICC" = "none" -a ! "${MPICXX}" = "none"
        then
          if test "$MPI_RUN" = "none"
          then AC_MSG_ERROR([MPI enabled but MPI_RUN not defined and no mpirun found])
          fi
          #
          # open-mpi and lam both support 'mpicc -showme'
          #
          if $MPICXX -showme > /dev/null 2>&1
          then
            MPI_CPPFLAGS="`$MPICXX -showme:compile`"
            MPI_LIBS="`$MPICXX -showme:link`"

            # This may be open-mpi or lam
            save_CPPFLAGS="$CPPFLAGS"
            CPPFLAGS="$CPPFLAGS $MPI_CPPFLAGS"
            AC_CHECK_DECL([OPEN_MPI], [MPI_BACKEND=openmpi],,[#include <mpi.h>])
            CPPFLAGS="$save_CPPFLAGS"
            if test "$MPI_BACKEND" = "openmpi"
            then
	      MPI_CPPFLAGS="-DOMPI_SKIP_MPICXX"
              # Unfortunately, only open-mpi allows us to reliably use the MPI C API.
	      for d in `$MPICC -showme:incdirs`; do
	        MPI_CPPFLAGS="$MPI_CPPFLAGS -I$d"
              done
              MPI_LIBS="`$MPICC -showme:link`"
            fi
          #
          # mpich2 and intelmpi both support 'mpicc -show'
          #
          elif $MPICXX -show > /dev/null 2>&1
          then
            # Intel MPI looks like MPICH, except that 'mpicxx -show' emits an
            # extra command to check that the compiler is setup properly, which
            # confuses our option extraction below.  We use '-nocompchk' to
            # disable this command.
            if test "$enable_mpi" = "intelmpi"
            then 
              command="$MPICXX -nocompchk -show -c | cut -d' ' -f2- | sed -e 's/ -c / /'"
              compile_options="`eval $command`"
              command="$MPICXX -nocompchk -show | cut -d' ' -f2-"
              link_options="`eval $command`"
            else 
              command="$MPICXX -show -c | cut -d' ' -f2- | sed -e 's/ -c / /'"
              compile_options="`eval $command`"
              command="$MPICXX -show | cut -d' ' -f2-"
              link_options="`eval $command`"
            fi
            # We need to do some dance to filter out the CPPFLAGS and LIBS.
            MPI_CPPFLAGS="$compile_options"
            # Just filter out the options we know aren't used during linking.
            # @<:@ and @:>@ are quadrigraphs representing [ and ] respectively.
            command="echo ' ' $link_options | \
                       sed -e 's/ -@<:@DUI@:>@@<:@ \t@:>@*@<:@^ \t@:>@*//'"
            MPI_LIBS="`eval $command`"
          else
            AC_MSG_WARN([Unable to invoke ${MPICXX}])
          fi
        else
          AC_MSG_RESULT([No MPICC found])
        fi

        save_CPPFLAGS="$CPPFLAGS"
        CPPFLAGS="$CPPFLAGS $MPI_CPPFLAGS"
        save_LIBS="$LIBS"
        LIBS="$LIBS $MPI_LIBS"

        # Now test whether it is mpich2 or intelmpi (both define the same macros)
        if test "$MPI_BACKEND" = "none";
        then 
          AC_CHECK_DECL([MPICH2], [MPI_BACKEND=mpich2],,[#include <mpi.h>])
          if test "$enable_mpi" = "intelmpi" -a "$MPI_BACKEND" = "mpich2"
          then MPI_BACKEND=intelmpi
          fi
        fi
        # Make sure we found what we were looking for:
        if test "$enable_mpi" != "probe" -a "$enable_mpi" != "yes" -a \
                "$enable_mpi" != "$MPI_BACKEND";
        then AC_MSG_ERROR([$enable_mpi requested, but $MPI_BACKEND found.])
        fi

        if test "$MPI_BACKEND" != "none"
        then
          # Now link the following to see whether the MPI_INIT() can be resolved.
          # (We may not be able to run it without first starting an mpi demon, so only link.)
          AC_MSG_CHECKING([for MPI libs])
          AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([#include <mpi.h>],[MPI_Init(0, 0);])],
            [AC_MSG_RESULT(found)],
            [AC_MSG_ERROR([Unable to compile / link test MPI application.])])
        fi
        CPPFLAGS="$save_CPPFLAGS"
        LIBS="$save_LIBS"
      ;;

      *)
        AC_MSG_ERROR([Unknown MPI library $enable_mpi])
      ;;
    esac
  fi
  ############################################################################
  # Second step: Test the found compiler flags and set output variables.
  ############################################################################

  # Find the applet names to boot / halt the MPI service.
  case "$MPI_BACKEND" in
    mpich2)
      AC_PATH_PROGS(MPI_BOOT, mpdboot,, $mpi_path)
      AC_PATH_PROGS(MPI_HALT, mpdcleanup,, $mpi_path)
    ;;
  esac
  AC_SUBST(MPI_BOOT)
  AC_SUBST(MPI_HALT)
  AC_SUBST(MPI_RUN)

  if test "$MPI_BACKEND" != "none"; then
    AC_SUBST(OVXX_HAVE_MPI, 1)
    AC_DEFINE_UNQUOTED(OVXX_HAVE_MPI, 1, [Define if MPI is available.])
  fi


  AC_SUBST(MPI_CPPFLAGS)
  AC_SUBST(MPI_LIBS)
  AC_SUBST(MPI_BACKEND)

fi

if test "$enable_mpi" = "probe" -o "$enable_mpi" = "yes"
then
  if test "$MPI_BACKEND" = "none"
  then mpi_backend="probe -- not found"
  else mpi_backend="probe -- found ($MPI_BACKEND)"
  fi
else
  mpi_backend=$enable_mpi
fi

])
