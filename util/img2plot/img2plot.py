import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio
import skimage.exposure
import skimage.draw
import svgwrite

# ---------------------------------------------------------------------------------------------------------------------
# Input and Output

INPUT_IMAGE_PATH = "path/to/input/file.png"
OUTPUT_SVG_PATH = "path/to/results/file.svg"

# ---------------------------------------------------------------------------------------------------------------------
# Configuration parameters. These affect the appearance of the output image.

# The program will continue drawing lines on the highest-intensity edges until the max intensity value drops below
# this fraction of the initial peak value. A smaller number means more lines will be drawn.
TERMINATION_RATIO = 1.0 / 3.5

# A line is extended until the edge intensity drops below this fraction of the corresponding peak edge intensity.
# Larger values mean many small lines; smaller values cause lines to extend for longer distances.
LINE_CONTINUE_THRESH = 0.01

# Lines must be longer than this length in pixels, else they will not be drawn.
MIN_LINE_LENGTH = 21

# Sets the amount of angle change before a line will be terminated.
MAX_CURVE_ANGLE_DEG = 20.0

# When drawing/extending lines, each new pixel contributes to the line direction, via a low‐pass filter with this
# attack value. Higher numbers mean lines can turn faster.
LPF_ATK = 0.05

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the monochrome image as a preprocessing step.
USE_CLAHE = True
CLAHE_KERNEL_SIZE = 32

# Apply a Gaussian blur as a preprocessing step.
USE_GAUSSIAN_BLUR = True
GAUSSIAN_KERNEL_SIZE = 1

# ---------------------------------------------------------------------------------------------------------------------
# Utility functions

def rgb2gray(rgb):
    """Convert an RGB image to grayscale using luminosity weights."""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def bilinearInterpolate(img, coords):
    """Perform bilinear interpolation for subpixel coordinates (x, y) in an image."""
    x, y = coords
    xfloat = x - math.floor(x)
    yfloat = y - math.floor(y)

    xfloor = math.floor(x)
    yfloor = math.floor(y)
    xceil = math.ceil(x)
    yceil = math.ceil(y)

    if xfloor < 0:
        xfloor = 0
    if xceil >= img.shape[1]:
        xceil = img.shape[1] - 1

    if yfloor < 0:
        yfloor = 0
    if yceil >= img.shape[0]:
        yceil = img.shape[0] - 1

    topLeft = img[int(yfloor), int(xfloor)]
    topRight = img[int(yfloor), int(xceil)]
    bottomLeft = img[int(yceil), int(xfloor)]
    bottomRight = img[int(yceil), int(xceil)]

    topMid = xfloat * topRight + (1 - xfloat) * topLeft
    botMid = xfloat * bottomRight + (1 - xfloat) * bottomLeft

    mid = yfloat * botMid + (1 - yfloat) * topMid

    return mid

def getLineFromGradient(img, point, gradients):
    """
    Attempt to grow a line from the given point following the gradient direction.
    Returns: (start_x, start_y, end_x, end_y, total_length)
    """
    px, py = point
    gradx, grady = gradients
    angle = math.atan2(grady[py, px], gradx[py, px])

    len_left = 0
    len_right = 0

    startx = px
    starty = py
    endx = px
    endy = py

    mangle = angle

    # Grow the "start" side.
    while (0 < starty < img.shape[0] - 1 and 0 < startx < img.shape[1] - 1 and
           bilinearInterpolate(img, (startx, starty)) > LINE_CONTINUE_THRESH * img[py, px]):

        len_left += 1

        # Recalculate angle to allow the line to "follow" curves.
        cangle = math.atan2(grady[int(round(starty)), int(round(startx))],
                            gradx[int(round(starty)), int(round(startx))])

        # Low-pass filtered angle update.
        mangle = mangle * (1 - LPF_ATK) + cangle * LPF_ATK

        if abs(angle - mangle) > MAX_CURVE_ANGLE_DEG * (2 * math.pi / 360):
            break

        startx = px + len_left * math.sin(mangle)
        starty = py - len_left * math.cos(mangle)

    mangle = angle

    # Grow the "end" side.
    while (0 < endy < img.shape[0] - 1 and 0 < endx < img.shape[1] - 1 and
           bilinearInterpolate(img, (endx, endy)) > LINE_CONTINUE_THRESH * img[py, px]):

        len_right += 1

        cangle = math.atan2(grady[int(round(endy)), int(round(endx))],
                            gradx[int(round(endy)), int(round(endx))])

        mangle = mangle * (1 - LPF_ATK) + cangle * LPF_ATK

        if abs(angle - mangle) > MAX_CURVE_ANGLE_DEG * (2 * math.pi / 360):
            break

        endx = px - len_right * math.sin(mangle)
        endy = py + len_right * math.cos(mangle)

    total_length = (len_left + len_right + 1)
    return int(round(startx)), int(round(starty)), int(round(endx)), int(round(endy)), total_length

# ---------------------------------------------------------------------------------------------------------------------
# Main processing

# Create the SVG drawing.
dwg = svgwrite.Drawing(OUTPUT_SVG_PATH, profile='tiny')

# Read and process the base image.
baseImage = imageio.imread(INPUT_IMAGE_PATH)
baseImageGray = rgb2gray(baseImage)

# Normalize to 0..1.
normImgGray = baseImageGray - baseImageGray.min()
normImgGray = normImgGray / normImgGray.max()

# Optionally apply CLAHE to bring out details.
if USE_CLAHE:
    normImgGray = skimage.exposure.equalize_adapthist(normImgGray, kernel_size=CLAHE_KERNEL_SIZE)

# Optionally apply a Gaussian blur to reduce noise.
if USE_GAUSSIAN_BLUR:
    normImgGray = ndimage.gaussian_filter(normImgGray, GAUSSIAN_KERNEL_SIZE)

plt.imshow(normImgGray, cmap='gray')
plt.title("Normalized Grayscale Image")
plt.show()

# Compute Sobel gradients.
sobelDx = ndimage.sobel(normImgGray, axis=0)  # horizontal derivative
sobelDy = ndimage.sobel(normImgGray, axis=1)  # vertical derivative
mag = np.hypot(sobelDx, sobelDy)

# Increase the probability of drawing a line where the image is locally darker.
imgBlur = ndimage.gaussian_filter(normImgGray, 2)
mag = np.multiply(mag, imgBlur.max() - imgBlur)

# Turn 'mag' into a probability distribution.
mag = mag / np.sum(mag)

plt.imshow(mag)
plt.title("Edge Magnitude (Probability Distribution)")
plt.colorbar()
plt.show()

# Compute gradients of the normalized image.
magGradY, magGradX = np.gradient(normImgGray)

plt.imshow(magGradX)
plt.title("Gradient X")
plt.show()

plt.imshow(magGradY)
plt.title("Gradient Y")
plt.show()

# Prepare images for drawing lines.
lineImg = np.zeros(mag.shape) - 1  # (unused in later code)
outImg = np.zeros(mag.shape, dtype=np.uint8)

# Statistics and control parameters.
initmaxp = mag.max()
cmax = initmaxp
i = 0

llacc = 0.0  # accumulator for mean line length
llcnt = 0.0  # line count
minll = 1e9
maxll = 0

while cmax > initmaxp * TERMINATION_RATIO:
    i += 1
    if i % 250 == 0:
        print("Max P:", mag.max(), "term at:", initmaxp * TERMINATION_RATIO)
        print("Line Stats: N=", llcnt, "length: min", minll, "mean", (llacc / llcnt if llcnt else 0), "max", maxll)
        llacc = 0
        llcnt = 0
        minll = 99999
        maxll = 0

    # Find the pixel with the maximum value in the probability distribution.
    pixIdx = np.argmax(mag)
    pIdxRow = pixIdx // mag.shape[1]
    pIdxCol = pixIdx % mag.shape[1]
    cmax = mag[pIdxRow, pIdxCol]

    # Get the line extending from this edge point.
    lstartx, lstarty, lendx, lendy, totalLength = getLineFromGradient(
        mag, (pIdxCol, pIdxRow), (magGradX, magGradY)
    )

    if totalLength < MIN_LINE_LENGTH:
        # This line is too short. Instead of drawing it, we replace its peak with the average of its neighbors.
        acc = 0.0
        cnt = 0

        if pIdxRow + 1 < mag.shape[0]:
            acc += mag[pIdxRow + 1, pIdxCol]
            cnt += 1
        if pIdxCol + 1 < mag.shape[1]:
            acc += mag[pIdxRow, pIdxCol + 1]
            cnt += 1
        if pIdxRow - 1 >= 0:
            acc += mag[pIdxRow - 1, pIdxCol]
            cnt += 1
        if pIdxCol - 1 >= 0:
            acc += mag[pIdxRow, pIdxCol - 1]
            cnt += 1

        mag[pIdxRow, pIdxCol] = acc / cnt if cnt > 0 else 0
        continue

    # Draw the line in the SVG image.
    dwg.add(dwg.line((lstartx, lstarty), (lendx, lendy), stroke=svgwrite.rgb(0, 0, 0, '%')))

    # Collect line statistics.
    llacc += totalLength
    llcnt += 1
    minll = min(minll, totalLength)
    maxll = max(maxll, totalLength)

    # Draw the line for preview purposes.
    rr, cc, val = skimage.draw.line_aa(lstarty, lstartx, lendy, lendx)
    rrd, ccd = skimage.draw.line(lstarty, lstartx, lendy, lendx)

    # Clip indices to image bounds.
    rr = np.clip(rr, 0, mag.shape[0] - 1)
    cc = np.clip(cc, 0, mag.shape[1] - 1)
    rrd = np.clip(rrd, 0, mag.shape[0] - 1)
    ccd = np.clip(ccd, 0, mag.shape[1] - 1)

    # Draw the line in the preview image.
    outImg[rrd, ccd] = 255

    # Remove the line's pixels from the edge magnitude image.
    mag[rr, cc] = 0
    mag[pIdxRow, pIdxCol] = 0  # Also knock down the peak that created this line.

# Finalize the output preview image.
outImg = np.clip(outImg, 0, 255)
outImg = -1 * outImg + 255  # Invert the image for display.

# Save the SVG file.
dwg.save()
print("SVG file saved to", OUTPUT_SVG_PATH)

plt.imshow(outImg, cmap='gray')
plt.title("Output Preview Image")
plt.show()

