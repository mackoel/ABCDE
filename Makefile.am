## Process this file with automake to produce Makefile.in
## Created by Anjuta

if ENABLE_OPENMPI_BUILD_TARGET
MPI_BUILD_TARGET = abcde_openmpi
else
MPI_BUILD_TARGET =
endif

if ENABLE_MPI_BUILD_TARGET
MPI_BUILD_TARGET = abcde_mpi
else
MPI_BUILD_TARGET =
endif

SUBDIRS = po

dist_doc_DATA = \
	README \
	COPYING \
	AUTHORS \
	ChangeLog \
	INSTALL \
	NEWS

INTLTOOL_FILES = intltool-extract.in \
	intltool-merge.in \
	intltool-update.in

AM_CPPFLAGS = \
	-DPACKAGE_LOCALE_DIR=\""$(localedir)"\" \
	-DPACKAGE_SRC_DIR=\""$(srcdir)"\" \
	-DPACKAGE_DATA_DIR=\""$(pkgdatadir)"\" \
	-IABCDE_SSM \
	$(ABCDE_CFLAGS)

AM_CFLAGS =\
         -Wall\
         -g

bin_PROGRAMS = \
	abcde $(MPI_BUILD_TARGET)

abcde_SOURCES = \
        ABCDE_SSM/abcde.cpp \
	ABCDE_SSM/deep.cpp \
	ABCDE_SSM/distr.cpp \
	ABCDE_SSM/main.cpp \
	ABCDE_SSM/solution.cpp \
	ABCDE_SSM/run_manager.cpp

abcde_HEADERS = \
	ABCDE_SSM/abcde.h \
	ABCDE_SSM/config.h \
	ABCDE_SSM/deep.h \
	ABCDE_SSM/distr.h \
	ABCDE_SSM/model.h \
	ABCDE_SSM/pch.h \
	ABCDE_SSM/solution.h \
	ABCDE_SSM/run_manager.h

abcde_openmpi_SOURCES = $(abcde_SOURCES)

abcde_openmpi_HEADERS = $(abcde_HEADERS)

abcde_mpi_SOURCES = $(abcde_SOURCES)

abcde_mpi_HEADERS = $(abcde_HEADERS)

abcdedir = \
        $(includedir)/abcde

abcde_openmpidir = $(abcdedir)

abcde_mpidir = $(abcdedir)

abcde_CXXFLAGS = -fPIC \
		-fopenmp \
		--std=c++11

abcde_CFLAGS = $(abcde_CXXFLAGS) 

abcde_LDFLAGS = -fPIC

abcde_openmpi_CXXFLAGS = -DMPIZE \
		$(abcde_CXXFLAGS) \
		$(MPI_CFLAGS)

abcde_openmpi_CFLAGS = $(abcde_openmpi_CXXFLAGS) 

abcde_openmpi_LDFLAGS = $(abcde_LDFLAGS)

abcde_mpi_CXXFLAGS = -DMPIZE \
		$(abcde_CXXFLAGS)

abcde_mpi_CFLAGS = $(abcde_mpi_CXXFLAGS) 

abcde_mpi_LDFLAGS = $(abcde_LDFLAGS)

abcde_LDADD = $(ABCDE_LIBS) \
	-lboost_filesystem \
	-lboost_regex \
	-lboost_system \
	-lboost_program_options

abcde_openmpi_LDADD = $(MPI_CLIBS) \
	$(abcde_LDADD)

abcde_mpi_LDADD = $(abcde_LDADD)

if NATIVE_WIN32
abcde_LDFLAGS += -mwindows
endif

EXTRA_DIST = \
	$(INTLTOOL_FILES)

DISTCLEANFILES = intltool-extract \
	intltool-merge \
	intltool-update \
	po/.intltool-merge-cache


# Remove doc directory on uninstall
uninstall-local:
	-rm -r $(docdir)
