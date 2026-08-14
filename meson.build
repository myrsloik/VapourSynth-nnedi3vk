project('nnedi3vk', 'cpp',
    default_options: ['buildtype=release', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++20'],
    license: 'GPL-3.0-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.2.3',
    version: '2.0',
)

deps = [
    dependency('vulkan-headers'),
]

incdir = include_directories(run_command('vapoursynth', 'get-include', check: true).stdout().strip())

py = import('python').find_installation(pure: false)

# The kernels ship as source and are compiled by the core at filter creation
# (VSVULKANAPI::compileGPUShader, which caches per core by source text). This step only turns
# them into string literals: it resolves the common.glsl include the core's compiler has no
# handler for and strips #version so a #define PIXEL_TYPE preamble can precede the body. No
# glslc, so building the plugin needs no shader toolchain at all -- the tradeoff is that the
# core's compiler runs no optimizer, where glslc previously ran -O.
shaders = custom_target(
    command: [py, files('src/embed_shaders.py'), '@OUTPUT@', '@INPUT@'],
    input: files(
        'src/shaders/common.glsl',
        'src/shaders/pad.comp',
        'src/shaders/prescreen.comp',
        'src/shaders/predict.comp',
        # The cooperative matrix predict variant, selected at runtime in place of the fp16
        # kernel when the device advertises the tile it is written against; see the shape probe
        # in nnedi3vk.cpp.
        'src/shaders/predict_cm.comp',
        'src/shaders/rowcopy.comp',
    ),
    output: 'shadersources.h',
)

shared_module('nnedi3vk',
    files('src/nnedi3vk.cpp'),
    shaders,
    dependencies: deps,
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins/nnedi3vk',
    install_rpath: '/opt/homebrew/lib:/usr/local/lib',
    name_prefix: '',
)

install_data('nnedi3_weights.bin', install_dir: py.get_install_dir() / 'vapoursynth/plugins/nnedi3vk')
