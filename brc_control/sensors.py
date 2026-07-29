import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

X = 1
Y = 0

class VirtualDistanceSensor:
    def __init__(self, pixelSize=218, meterSize=30.0, numDistances=16):
        self.pixelSize = pixelSize
        self.meterSize = meterSize
        self.numDistances = numDistances
        self.numDistHalf = int(numDistances / 2)
        self.metersPerPixel = meterSize / pixelSize
        self.centerX = pixelSize * 0.5
        self.wh = (pixelSize, pixelSize)
        self.mpp = (self.metersPerPixel, self.metersPerPixel)
        self.centerMetersX = meterSize * 0.5
        self.centerMetersY = meterSize
        self.angles = np.arange(180, step=180 / self.numDistances)
        self.rayLinesX = []
        self.rayLinesY = []

        self.__createDistanceMatrix()
        self.__createRays()

    '''
    euclidean distance in meters
    creates a 2d-matrix of image size and calculates euclidean distance for each pixel
    '''
    def __createDistanceMatrix(self):
        self.indices = np.indices(self.wh)
        self.distances = np.sqrt(np.square(self.indices[X,:] * self.mpp[X] - self.centerX * self.mpp[X]) + np.square(self.indices[Y,:] * self.mpp[Y] - self.wh[Y] * self.mpp[Y]))
        self.maxDistance = np.max(self.distances)

        # reshape distances array to 3rd dimension with length of numDistances
        # it copies the array along the new axis
        # source array needs to be expanded with np.newaxis or np.expand_dims
        self.expandedDistances = np.broadcast_to(self.distances[:,:,np.newaxis], self.distances.shape+(self.numDistances,))
        self.lineMasks = np.empty((self.wh[Y], self.wh[X], self.numDistances))

        # init for faster numpy version:
        self.maxArray = np.full(self.expandedDistances.shape, self.maxDistance)

    def __createRays(self):
        for i in range(self.numDistances):
            # commented: offset for rays to accomodate for vehicle
            # TODO check if center ray has to be copied and shifted too
            # 1.0 at drTemp.line is the offset in meters
            #direction = -1.0
            #if self.angles[i] > 90:
            #    direction = 1.0
            #elif self.angles[i] == 90:
            #    direction = 0.0
            x = self.centerMetersX + self.maxDistance * math.cos(math.radians(self.angles[i] + 180))
            x = x / self.mpp[X]
            y = self.centerMetersY + self.maxDistance * math.sin(math.radians(self.angles[i] + 180))
            y = y / self.mpp[Y]
            self.rayLinesX.append(x)
            self.rayLinesY.append(y)

            imgTemp = Image.fromarray(np.zeros((self.wh[Y], self.wh[X])).astype(np.uint8))
            drTemp = ImageDraw.Draw(imgTemp)
            drTemp.line((self.centerX, self.wh[Y]) + (x, y), fill=255)
            #drTemp.line((self.centerX + (direction * 1.0 / self.mpp[X]), self.wh[Y]) + (x, y), fill=255)
            self.lineMasks[:,:,i] = (np.asarray(imgTemp) > 0)

        # convert to mask
        self.lineMasks = (self.lineMasks > 0)

    '''
    Given an image get the ray distances
    Propably should not be called from different threads
    '''
    def getDistances(self, imagePath='', image=None):
        if image is None:
            ground = Image.open(imagePath).resize(self.wh[::-1])
            groundMask = (np.asarray(ground)[:,:,0] == 7)
        else:
            groundMask = (np.asarray(image) > 0.5)
        groundMaskEdgesImg = Image.fromarray((groundMask * 255).astype(np.uint8)).filter(ImageFilter.CONTOUR) # FIND_EDGES
        groundMaskEdges = (np.asarray(groundMaskEdgesImg) == 0)
        groundMaskEdges[self.wh[Y]-2:,:] = False

        intersections = (self.lineMasks & np.expand_dims(groundMaskEdges, axis=2))
        #intersectionImg = Image.fromarray((intersections[:,:,self.numDistHalf] * 255).astype(np.uint8))

        self.maxArray[intersections] = self.expandedDistances[intersections]
        minsNp = np.min(self.maxArray, axis=(0,1))
        self.maxArray[intersections] = self.maxDistance
        return(list(minsNp[:]))

    '''
    Draws an image of the distances and sensor-rays
    '''
    def drawSensorImage(self):
        img = Image.fromarray(((self.distances / self.maxDistance) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(img)
        for i in range(self.numDistances):
            dr.line((self.centerX, self.wh[Y]) + (self.rayLinesX[i], self.rayLinesY[i]), fill=255)
        img.show()

    '''
    Visualize rays of the sensor given an array of distances
    The distance-array must have a length numDistances
    TODO: update ray generation to remove first ray as it lies on the X-axis
    '''
    def drawDistances(self, distances, imagePath=None, show=False):
        imgVis = Image.fromarray(np.zeros((self.wh[Y], self.wh[X])).astype(np.uint8))
        if imagePath is not None:
            imgVis = Image.open(imagePath).resize(self.wh[::-1])
            imgVis = imgVis.point(lambda i: i * 15.0 if i == 7 else i)
        drVis = ImageDraw.Draw(imgVis)
        for i in range(self.numDistances):
            x = self.centerMetersX + distances[i] * math.cos(math.radians(self.angles[i] + 180))
            x = x / self.mpp[X]
            y = self.centerMetersY + distances[i] * math.sin(math.radians(self.angles[i] + 180))
            y = y / self.mpp[Y]
            if i >= (int(self.numDistances / 2) - 2) and i <= (int(self.numDistances / 2) + 2): #anpassungsbedarf
                drVis.line((self.centerX, self.wh[Y]) + (x, y), fill=255)
        if show:
            imgVis.show()
        else:
            return imgVis
