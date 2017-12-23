warning_flags = -Wall -Wextra -Wshadow -Wno-unused-parameter
common_cflags = -O3 -ffast-math -fvisibility=hidden $(warning_flags) $(MFLAGS)
AM_CXXFLAGS = -std=c++14 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS) $(FFTW3F_CFLAGS)

lib_LTLIBRARIES = libdctfilter.la

libdctfilter_la_SOURCES = DCTFilter/DCTFilter.cpp

libdctfilter_la_LIBADD = $(FFTW3F_LIBS)

libdctfilter_la_LDFLAGS = -no-undefined -avoid-version $(PLUGINLDFLAGS)
