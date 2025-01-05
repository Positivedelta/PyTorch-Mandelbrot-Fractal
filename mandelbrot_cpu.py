#
# (c) Bit Parallel Ltd, January 2025
#

import torch
from torchvision.utils import save_image

# notes 1, the aspcect ratio of the complex range is 3.0 x 2.4, mapping to a default image size of 1280 x 1024
#       2, if you change this aspect ratio you'll need to update the complex number ranges, this all needs automating...
#
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 1024

def render(xMin = -2.2, xMax = 0.8, yMin = -1.2, yMax = 1.2, width = IMAGE_WIDTH, height = IMAGE_HEIGHT, maxIterations = 70):
    realStep = torch.linspace(xMin, xMax, width, dtype=torch.float64).cpu()
    imaginaryStep = torch.linspace(yMin, yMax, height, dtype=torch.float64).cpu()

    # create and initialise the complex 'c', 'z' and 'pixel' tensors
    # FIXME! decide which is faster, meshgrid or broadcasting? should be tested on a GPU using the PyTorch CUDA timing API
    #
#   cReal, cImaginary = torch.meshgrid([realStep, imaginaryStep], indexing = 'xy')
#   c = torch.complex(cReal, cImaginary)
#   c = realStep + (imaginaryStep * 1j).reshape(-1, 1)
    c = torch.complex(realStep, imaginaryStep.reshape(-1, 1)).cpu()
    z = torch.zeros(height, width, dtype=torch.cdouble).cpu()
    scale = torch.zeros(height, width, dtype=torch.float32).cpu()
    rgb = torch.zeros(3, height, width, dtype=torch.float32).cpu()

    for i in range(maxIterations):
        z = z * z + c

        # notes 1, using the complex conjugate to avoid invoking abs() as it includes an unneeded and expensive sqrt(), pytorch needs to add abs2()...
        #       2, z.imag * z.real introduces aritfacts in the fractal, no idea why...
        #       3, using smooth bernstein polynomials to generate the RGB components
        #
        scale[(rgb[0] == 0) & (rgb[1] == 0) & (rgb[2] == 0) & ((z * z.conj()).real > 4)] = i / maxIterations
        rgb[0] = 8.5 * (1.0 - scale) * (1.0 - scale) * (1.0 - scale) * scale
        rgb[1] = 15.0 * (1.0 - scale) * (1.0 - scale) * scale * scale
        rgb[2] = 9.0 * (1.0 - scale) * scale * scale * scale

    return rgb

def mandelbrot(x = -0.7, y = 0.0, zoom = 1.0, maxIterations = 70):
    xOffset = 1.5 / zoom
    xMin, xMax = x - xOffset, x + xOffset

    yOffset = 1.2 / zoom
    yMin, yMax = y + yOffset, y - yOffset

    return render(xMin, xMax, yMin, yMax, IMAGE_WIDTH, IMAGE_HEIGHT, maxIterations)

#
# other 'interesting' coordinates...
#
#fractal = mandelbrot()
#fractal = mandelbrot(-4.601222e-1, 5.702860e-1, 2.0e5, 384)
#fractal = mandelbrot(-1.186592, -1.901211e-1, 6.290223e+3, 384)
#

fractal = mandelbrot(-0.11, 1.0, 9.0, 70)
save_image(fractal, 'fractal.png')
