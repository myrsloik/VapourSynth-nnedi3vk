project('nnedi3vk', 'cpp',
    default_options: ['buildtype=release', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++20'],
    license: 'GPL-3.0-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.2.3',
    version: '1.0',
)

deps = [
    dependency('vulkan-headers'),
    dependency('vulkan-memory-allocator'),
    dependency('vulkan-volk'),
]

glslc = find_program('glslc')

incdir = include_directories(run_command('vapoursynth', 'get-include', check: true).stdout().strip())

py = import('python').find_installation(pure: false)

shaders = []

foreach kernel : ['prescreen', 'predict', 'predict_cv']
    foreach pixel : [['u8', '0'], ['u16', '1'], ['f16', '2'], ['f32', '3']]
        shaders += custom_target(
            command: [
                glslc,
                '-DPIXEL_TYPE=' + pixel[1],
                '-mfmt=num',
                '-O',
                '--target-env=vulkan1.4',
                '-o',
                '@OUTPUT@',
                '@INPUT@',
            ],
            depend_files: files('src/shaders/common.glsl'),
            input: files('src/shaders' / kernel + '.comp'),
            output: kernel + '_' + pixel[0] + '.h',
        )
    endforeach
endforeach

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
