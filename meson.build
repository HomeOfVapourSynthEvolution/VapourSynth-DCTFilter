project('DCTFilter', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++14'],
  meson_version : '>=0.49.0',
  version : '2'
)

sources = 'DCTFilter/DCTFilter.cpp'

compiler = meson.get_compiler('cpp')

if compiler.get_argument_syntax() == 'msvc'
  add_project_arguments('/fp:fast', language : 'cpp')
  deps = [ dependency('fftwf') ]
  install_dir = 'installed'  # dummy
else
  add_project_arguments('-ffast-math', language : 'cpp')
  if host_machine.cpu_family().startswith('x86')
    add_project_arguments('-mfpmath=sse', '-msse2', language : 'cpp')
  endif
  vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)
  deps = [ dependency('fftw3f'), vapoursynth_dep ]
  install_dir = join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth')
endif


shared_module('dctfilter', sources,
  dependencies : deps,
  install : true,
  install_dir : install_dir,
  gnu_symbol_visibility : 'hidden'
)
