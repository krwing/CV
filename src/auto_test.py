import sys, random, argparse
from PIL import Image, ImageDraw


def createRandomPixel(dims):#Fill a usable Pixel image with pixels, and then offset the pixels based on depthmap to achieve the effect of magic eye
    img = Image.new('RGB', dims)
    draw = ImageDraw.Draw(img)
    r = int(min(*dims)/100)
    #print("r:", r)
    n = 1000
    for i in range(n):
        x, y = random.randint(r, dims[0]-r), random.randint(r, dims[1]-r)
        fill = (random.randint(0, 255), random.randint(0, 255),
                random.randint(0, 255))
        draw.ellipse((x-r, y-r, x+r, y+r), fill)

    return img

def createPixelImage(tile, dims):#Expanding pixel images to the size of depth maps
    img = Image.new('RGB', dims)
    W, H = dims
    w, h = tile.size
    cols = int(W/w) + 1
    rows = int(H/h) + 1
    for i in range(rows):
        for j in range(cols):
            img.paste(tile, (j*w, i*h))

    return img


def createAutostereogram(dmap):#Move pixels based on depthmap to simulate spatial changes in depthmap
    if dmap.mode != 'L':
        dmap = dmap.convert('L')
    pixel = createRandomPixel((100, 100))#Randomly create a new pixel file
    img = createPixelImage(pixel, dmap.size)#Expand the size of pixel files to depthmap
    sImg = img.copy()
    pixD = dmap.load()
    pixS = sImg.load()
    cols, rows = sImg.size
    for j in range(rows):
        for i in range(cols):
            xshift = pixD[i, j]/10#Xshift is the degree of pixel offset at position (i, j), and the greater the depth at a position in the depth map, the greater the xshift
            xpos = i - pixel.size[0] + xshift
            if xpos > 0 and xpos < cols:
                pixS[i, j] = pixS[xpos, j]
    return sImg


def main():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument('--depth', dest='dmFile', required=False, default="depthmap.jpg")
    args = parser.parse_args()
    outFile = 'as.png'
    dmImg = Image.open(args.dmFile)
    asImg = createAutostereogram(dmImg)
    asImg.save(outFile)


if __name__ == '__main__':

    main()