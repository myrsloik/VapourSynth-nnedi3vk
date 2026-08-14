# nnedi3vk

NNEDI3 is an intra-field only deinterlacer. It takes in a frame, throws away one field, and then interpolates the missing pixels using only information from the kept field. It has same rate and double rate modes. NNEDI3 is also very good for enlarging images by powers of 2.

Runs on the VapourSynth core's Vulkan device: frames go in and come out GPU resident (`vnode:gpu`), so chains of GPU filters never leave VRAM. Feeding a CPU clip automatically inserts an upload, and `std.GPUDownload` brings the result back when the chain ends. The device is selected core-wide with `core.set_vulkan_device(index)` (see `core.vulkan_devices`) before first use.


## Requirement

VapourSynth R73+ with GPU support (API 4.3), and a Vulkan 1.4 capable GPU and driver.


## Usage

```py
nnedi3vk.NNEDI3(vnode:gpu clip, int field[, bint dh=False, int[] planes=[0, 1, 2], int nsize=6, int nns=1, int qual=1, int etype=0, int pscrn=2, bint coop_mat])
```

- clip: Clip to process. Any format with either 8-16 bit integer or 16/32 bit float is supported.

- field: Controls the mode of operation (double vs same rate) and which field is kept.
  - 0 = same rate, keep bottom field
  - 1 = same rate, keep top field
  - 2 = double rate (alternates each frame), starts with bottom
  - 3 = double rate (alternates each frame), starts with top

- dh: Doubles the height of the input. Each line of the input is copied to every other line of the output and the missing lines are interpolated. If `field=0`, the input is copied to the odd lines of the output. If `field=1`, the input is copied to the even lines of the output. `field` must be set to either 0 or 1 when using `dh=True`.

- planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied in `dh=False`, but will contain uninitialized memory in `dh=True`.

- nsize: Specifies the size of the local neighborhood around each pixel that is used by the predictor neural network. For image enlargement it is recommended to use 0 or 4. Larger y_diameter settings will result in sharper output. For deinterlacing larger x_diameter settings will allow connecting lines of smaller slope. However, what setting to use really depends on the amount of aliasing (lost information) in the source. If the source was heavily low-pass filtered before interlacing then aliasing will be low and a large x_diameter setting won't be needed, and vice versa.
  - 0 =  8x6
  - 1 = 16x6
  - 2 = 32x6
  - 3 = 48x6
  - 4 =  8x4
  - 5 = 16x4
  - 6 = 32x4

- nns: Specifies the number of neurons in the predictor neural network. 0 is fastest. 4 is slowest, but should give the best quality. This is a quality vs speed option; however, differences are usually small. The difference in speed will become larger as 'qual' is increased.
  - 0 = 16
  - 1 = 32
  - 2 = 64
  - 3 = 128
  - 4 = 256

- qual: Controls the number of different neural network predictions that are blended together to compute the final output value. Each neural network was trained on a different set of training data. Blending the results of these different networks improves generalization to unseen data. Possible values are 1 or 2. Essentially this is a quality vs speed option. Larger values will result in more processing time, but should give better results. However, the difference is usually pretty small. I would recommend using `qual>1` for things like single image enlargement.

- etype: Controls which set of weights to use in the predictor nn.
  - 0 = weights trained to minimize absolute error
  - 1 = weights trained to minimize squared error

- pscrn: Controls whether or not the prescreener neural network is used to decide which pixels should be processed by the predictor neural network and which can be handled by simple cubic interpolation. The prescreener is trained to know whether cubic interpolation will be sufficient for a pixel or whether it should be predicted by the predictor nn. The computational complexity of the prescreener nn is much less than that of the predictor nn. Since most pixels can be handled by cubic interpolation, using the prescreener generally results in much faster processing. The prescreener is pretty accurate, so the difference between using it and not using it is almost always unnoticeable. Higher levels for the new prescreener result in cubic interpolation being used on fewer pixels (so are slower, but incur less error). However, the difference is pretty much unnoticable. Level 2 is closest to the original prescreener in terms of incurred error, but is much faster.
  - 0 = no prescreening
  - 1 = original prescreener
  - 2 = new prescreener level 0
  - 3 = new prescreener level 1
  - 4 = new prescreener level 2

- coop_mat: Whether the predictor GEMM runs on cooperative matrix instructions (`VK_KHR_cooperative_matrix`) instead of the subgroup FMA kernel. Left unset it is chosen automatically: used when the core enabled the extension, the device advertises a 16x16x16 fp16->fp32 cooperative matrix at subgroup scope, and the clip is half float. Set True to require it -- an error is raised, naming the reason, if any of those does not hold. Set False to always use the subgroup FMA kernel.

  **Worth setting explicitly.** A device advertising the extension does not necessarily have matrix hardware: on RDNA2 (for example an RX 6900 XT) the driver emulates it on the ordinary vector ALU, and the automatic choice is then a slowdown of roughly 6-31% depending on `nsize`. The cooperative matrix tile pins the kernel to 16 pixels per subgroup where the FMA kernel picks 4, so it stages four times the shared memory and loses occupancy; with no matrix units to pay for that, it cannot win. Accuracy is very slightly better on the matrix path (fp32 accumulate from uncentered inputs, against fp16 accumulate with a centering correction), but the difference is far below the noise of the network itself.

Removed relative to the standalone (pre-GPU-node) versions: `device_index` and `list_device` (device selection is core-wide now), `coopvec` (the core device enables no vendor extensions; the FP32 subgroup GEMV path is used on all hardware), and `num_streams` (the core's execution pool sizes the in-flight bound from its worker threads and budgets the pinned memory itself).


## Installation

```
pip install -U vapoursynth-nnedi3vk
```
