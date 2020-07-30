dnl Process this file with autoconf to produce a configure script.
dnl Created by Anjuta application wizard.

AC_INIT(abcde, 0.1)

AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([1.11 subdir-objects])

dnl AM_SILENT_RULES([yes])

AC_PROG_CXX




dnl ***************************************************************************
dnl Internationalization
dnl ***************************************************************************
IT_PROG_INTLTOOL([0.35.0])

GETTEXT_PACKAGE=abcde
AC_SUBST(GETTEXT_PACKAGE)
AC_DEFINE_UNQUOTED(GETTEXT_PACKAGE,"$GETTEXT_PACKAGE", [GETTEXT package name])
AM_GLIB_GNU_GETTEXT



dnl ***************************************************************************
dnl Check for Windows
dnl ***************************************************************************
AC_CANONICAL_HOST

case $host_os in
  *mingw*)
    platform_win32=yes
    native_win32=yes
    ;;
  pw32* | *cygwin*)
    platform_win32=yes
    native_win32=no
    ;;
  *)
    platform_win32=no
    native_win32=no
    ;;
esac
AM_CONDITIONAL(PLATFORM_WIN32, test x"$platform_win32" = "xyes")
AM_CONDITIONAL(NATIVE_WIN32, test x"$native_win32" = "xyes")


LT_INIT([win32-dll])


dnl PKG_CHECK_MODULES(ABCDE, [glib-2.0 gobject-2.0 gmodule-2.0 gio-2.0 ])

# Define a macro that is used to parse a --with-mpi parameter
# The macro is named "CHECK_MPICC"
AC_DEFUN([CHECK_MPICC],[

        AC_ARG_WITH(
                openmpi,
                [  --with-openmpi[=DIR]     OpenMPI installation directory],
                ,
                [with_openmpi="/usr/lib64/openmpi"]
        )

        AC_MSG_CHECKING(for OpenMPI installation directory)

        if test "$with_openmpi" = "no"; then
                AC_MSG_ERROR( You need to specify the OpenMPI installation directory directory using --with-openmpi)
        else
		MPICXX="$with_openmpi/bin/mpicxx"
		MPILIBS="$with_openmpi/lib"
                if test -x $MPICC; then
                        MPI_CFLAGS=`$MPICXX --showme:compile 2>/dev/null`
                        MPI_LDADD=`$MPICXX --showme:link 2>/dev/null`
			MPI_CLIBS="-Wl,--rpath $MPILIBS $MPI_LDADD"
                        AC_MSG_RESULT(OpenMPI installation directory found!)
                else
                        AC_MSG_ERROR( $with_openmpi not found.  Check the value you specified with --with-openmpi)
                fi
        fi

])

dnl Let people enable/disable the OpenMPI stuff.
AC_ARG_ENABLE(openmpi-build-target,
              [  --enable-openmpi-build-target  Use openmpi-build-target to run on parallel machine [default=no]],
                enable_openmpi_build_target="$enableval", enable_openmpi_build_target=no)
if test x$enable_openmpi_build_target = xyes ; then
	# Now call the CHECK_MPICC macro that was just specified
	CHECK_MPICC
	AC_SUBST(MPI_CFLAGS)
	AC_SUBST(MPI_CLIBS)
fi
AM_CONDITIONAL(ENABLE_OPENMPI_BUILD_TARGET, test x$enable_openmpi_build_target = xyes)

dnl Let people enable/disable the OpenMPI stuff.
AC_ARG_ENABLE(mpi-build-target,
              [  --enable-mpi-build-target  Use mpi-build-target to run on parallel machine [default=no]],
                enable_mpi_build_target="$enableval", enable_mpi_build_target=no)
AM_CONDITIONAL(ENABLE_MPI_BUILD_TARGET, test x$enable_mpi_build_target = xyes)

AC_OUTPUT([
Makefile
po/Makefile.in
nsis/mingwbuild
nsis/ABCDE.nsis
])
