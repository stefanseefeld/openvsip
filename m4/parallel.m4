dnl Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved.
dnl
dnl File:   parallel.m4
dnl Author: Stefan Seefeld
dnl Date:   2007-12-20
dnl
dnl Contents: Parallel configuration for Sourcery VSIPL++
dnl

AC_DEFUN([SVXX_CHECK_PARALLEL],
[

# By default we will probe for MPI and use it if it exists.  If it
# does not exist, and no explicit MPI or PAS service is requested, 
# we will configure a serial VSIPL++ library.
#
# If the user specifies a parallel service either by an explicit
# --enable-parallel[=<service>] or by specifying the prefix to 
# MPI or PAS, then we search for MPI / PAS and issue an error if 
# it does not exist.
#
# If the user specifies that no parallel service should be used 
# (with --disable-parallel), then we do not search for it and 
# configure a serial VSIPL++ library.

AC_ARG_ENABLE([parallel],
  AS_HELP_STRING([--enable-parallel],
                 [Use Parallel service. Available backends are:
                  lam, mpich2, intelmpi, mpipro, and pas.
                  In addition, the value 'probe' causes configure to try.]),,
  [enable_parallel=probe])

AC_ARG_WITH(mpi_prefix,
  AS_HELP_STRING([--with-mpi-prefix=PATH],
                 [Specify the installation prefix of the MPI library.  Headers
                  must be in PATH/include; libraries in PATH/lib.]),
  dnl If the user specified --with-mpi-prefix, they mean to use MPI for sure.
  [
    if test -z $enable_parallel
    then enable_parallel=yes
    fi
  ])

AC_ARG_WITH(mpi_prefix64,
  AS_HELP_STRING([--with-mpi-prefix64=PATH],
                 [Specify the installation prefix of the MPI library.  Headers
                  must be in PATH/include64; libraries in PATH/lib64.]),
  dnl If the user specified --with-mpi-prefix64, they mean to use MPI for sure.
  [
    if test -z $enable_parallel
    then enable_parallel=yes
    fi
  ])

AC_ARG_WITH(mpi_cxxflags,
  AS_HELP_STRING([--with-mpi-cxxflags=FLAGS],
                 [Specify the C++ compiler flags used to compile with MPI.]))

AC_ARG_WITH(mpi_libs,
  AS_HELP_STRING([--with-mpi-libs=LIBS],
                 [Specify the linker library flags used to compile with MPI.]))

### Mercury PAS

AC_ARG_WITH(pas_include,
  AS_HELP_STRING([--with-pas-include=PATH],
                 [Specify the path to the PAS include directory.]),
  dnl If the user specified --with-pas-include, they mean to use PAS for sure.
  [enable_parallel=pas])

AC_ARG_WITH(pas_lib,
  AS_HELP_STRING([--with-pas-lib=PATH],
                 [Specify the installation path of the PAS library.]),
  dnl If the user specified --with-pas-lib, they mean to use PAS for sure.
  [enable_parallel=pas])

AC_ARG_ENABLE([pas_heap_size],
  AS_HELP_STRING([--enable-pas-heap-size=SIZE],
                 [Set PAS heap size.  Default is 0x100000]),,
  [enable_pas_heap_size=0x100000])

AC_ARG_ENABLE([pas_share_dynamic_xfer],
  AS_HELP_STRING([--enable-pas-share-dynamice-xfer],
                 [Share a PAS dynamic xfer object. Default is not to.]),,
  [enable_pas_share_dynamic_xfer=no])

# If the user specified an MPI prefix, they definitely want MPI.
# However, we need to avoid overwriting the value of $enable_mpi
# if the user set it (i.e. '--enable-mpi=mpipro').

if test -n "$with_mpi_prefix" -o -n "$with_mpi_prefix64"
then
  if test "$enable_parallel" == "no"
  then AC_MSG_RESULT([MPI disabled, but MPI prefix given.])
  elif test "$enable_parallel" == "probe"
  then enable_parallel="yes"
  fi
fi

MPI_CPPFLAGS=
MPI_LDFLAGS=
MPI_LIBS=
PAR_SERVICE=none
vsip_impl_avoid_posix_memalign=

if test "$enable_parallel" != "no"
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
    PAR_SERVICE=$enable_parallel
    
  else
    case "$enable_parallel" in
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

        if test ! "$MPICC" = "none" -a ! "${MPICXX}" = "none"
        then
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
            AC_CHECK_DECL([OPEN_MPI], [PAR_SERVICE=openmpi],,[#include <mpi.h>])
            CPPFLAGS="$save_CPPFLAGS"
            if test "$PAR_SERVICE" = "openmpi"
            then
              # Unfortunately, only open-mpi allows us to reliably use the MPI C API.
              MPI_CPPFLAGS="-DOMPI_SKIP_MPICXX `$MPICC -showme:compile`"
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
            if test "$enable_parallel" = "intelmpi"
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

        # If this is open-mpi, we already know (see above).
        # Now test whether it is lam
        if test "$PAR_SERVICE" = "none"; then
          AC_CHECK_DECL([LAM_MPI],
                        [PAR_SERVICE=lam
                         vsip_impl_avoid_posix_memalign=yes],,
                        [#include <mpi.h>])
        fi
        # Now test whether it is mpich2 or intelmpi (both define the same macros)
        if test "$PAR_SERVICE" = "none";
        then 
          AC_CHECK_DECL([MPICH2], [PAR_SERVICE=mpich2],,[#include <mpi.h>])
          if test "$enable_parallel" = "intelmpi" -a "$PAR_SERVICE" = "mpich2"
          then PAR_SERVICE=intelmpi
          fi
        fi
        # Make sure we found what we were looking for:
        if test "$enable_parallel" != "probe" -a "$enable_parallel" != "yes" -a \
                "$enable_parallel" != "$PAR_SERVICE";
        then AC_MSG_ERROR([$enable_parallel requested, but $PAR_SERVICE found.])
        fi

        if test "$PAR_SERVICE" != "none"
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

      # For PAS the user may have provide --with-pas-include et al.
      # Alternatively, we search for a pkg-config descriptor
      pas)
        save_CPPFLAGS="$CPPFLAGS"
        save_LDFLAGS="$LDFLAGS"
        save_LIBS="$LIBS"

        pas_found="no"
        try_pas="direct pkgconfig"

        for try in $try_pas
          do
          if test $try = "direct"
          then
            AC_MSG_CHECKING([for PAS (direct)])
            if test -n "$with_pas_include"
            then CPPFLAGS="$save_CPPFLAGS -I$with_pas_include"
            fi
            if test -n "$with_pas_lib"
            then LDFLAGS="$save_LDFLAGS -L$with_pas_lib"
            fi
            LIBS="$save_LIBS -lpas"
          elif test $try = "pkgconfig"
          then
            AC_MSG_CHECKING([for PAS (with pkg-config)])
            PKG_CHECK_MODULES(PAS, pas)
            CPPFLAGS="$save_CPPFLAGS $PAS_CFLAGS"
            LIBS="$save_LIBS $PAS_LIBS"
          else
            AC_MSG_ERROR([Unknown PAS try $try])
          fi

          AC_LINK_IFELSE(
          [
            AC_LANG_PROGRAM
            (
              [[#include <pas.h>]],
              [[PAS_id pset;pas_pset_close(pset, 0);]]
            )
          ],
          [pas_found=$try;AC_MSG_RESULT([found]);break],
          [pas_found="no";AC_MSG_RESULT([not found]) ]
          )
        done

        if test "$pas_found" == "no"
        then
          if test "$with_pas" != "probe"
          then AC_MSG_ERROR([PAS enabled but no library found])
          fi
          AC_MSG_RESULT([No PAS library found])
          CPPFLAGS=$save_CPPFLAGS
          LDFLAGS=$save_LDFLAGS
          LIBS=$save_LIBS
        else
          AC_MSG_RESULT([Using $pas_found for PAS])
          PAR_SERVICE=pas

          # These values are not used if PAS is not enabled (i.e. if PAR_SERVICE != 2).
          # They are always defined for binary packaging convenience.  This allows
          # the same acconfig.hpp to be used with/without PAS.

          AC_DEFINE_UNQUOTED(VSIP_IMPL_PAS_HEAP_SIZE, $enable_pas_heap_size,
            [Define the heap size used inside the PAS backend.])

          if test $enable_pas_share_dynamic_xfer = "yes"
          then enable_pas_share_dynamic_xfer=1
          else enable_pas_share_dynamic_xfer=0
          fi

          AC_DEFINE_UNQUOTED(VSIP_IMPL_PAS_SHARE_DYNAMIC_XFER,
            $enable_pas_share_dynamic_xfer,
            [Define to 1 to share a dynamic_xfer object, 0 otherwise.])
        fi
      ;;
      mpipro)
        # MPI/Pro does not have any identifying macros.
        # Require user to specify --enable-mpi=mpipro
        PAR_SERVICE=mpipro
        MPI_CPPFLAGS=""
        MPI_LIBS="-lmpipro"
        PAR_SERVICE="mpipro"
      ;;
      *)
        AC_MSG_ERROR([Unknown MPI library $enable_parallel])
      ;;
    esac
  fi
  ############################################################################
  # Second step: Test the found compiler flags and set output variables.
  ############################################################################

  # Find the applet names to boot / halt the parallel service.
  case "$PAR_SERVICE" in
    lam)
      AC_PATH_PROGS(PAR_BOOT, lamboot,, $mpi_path)
      AC_PATH_PROGS(PAR_HALT, lamhalt lamwipe wipe,, $mpi_path)
    ;;
    mpich2)
      AC_PATH_PROGS(PAR_BOOT, mpdboot,, $mpi_path)
      AC_PATH_PROGS(PAR_HALT, mpdcleanup,, $mpi_path)
    ;;
  esac
  AC_SUBST(PAR_BOOT)
  AC_SUBST(PAR_HALT)

  if test "$PAR_SERVICE" = "none"
  then vsipl_par_service=0
  elif test "$PAR_SERVICE" = "pas"
  then
    vsipl_par_service=2
    AC_SUBST(VSIP_IMPL_HAVE_PAS, 1)
  else
    # must be MPI
    vsipl_par_service=1
    AC_SUBST(VSIP_IMPL_HAVE_MPI, 1)
  fi

  if test "$neutral_acconfig" = 'y'
  then CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_PAR_SERVICE=$vsipl_par_service"
  else
    AC_DEFINE_UNQUOTED(VSIP_IMPL_PAR_SERVICE, $vsipl_par_service,
      [Define to parallel service provided (0 == no service, 1 = MPI, 2 = PAS).])
  fi

  CPPFLAGS="$CPPFLAGS $MPI_CPPFLAGS"
  LIBS="$LIBS $MPI_LIBS"
  AC_SUBST(PAR_SERVICE)

  if test -n "$vsip_impl_avoid_posix_memalign"
  then if test "$neutral_acconfig" = 'y'
  then CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_AVOID_POSIX_MEMALIGN=1"
  else AC_DEFINE_UNQUOTED(VSIP_IMPL_AVOID_POSIX_MEMALIGN, 1,
  [Set to 1 to avoid using posix_memalign (LAM defines its own malloc,
  including memalign but not posix_memalign).])
  fi; AC_MSG_NOTICE(
  [Avoiding posix_memalign, may not be compatible with LAM-MPI malloc])
  fi
fi

if test "$enable_parallel" = "probe" -o "$enable_parallel" = "yes"
then
  if test "$PAR_SERVICE" = "none"
  then par_service="probe -- not found"
  else par_service="probe -- found ($PAR_SERVICE)"
  fi
else
  par_service=$enable_parallel
fi

])
