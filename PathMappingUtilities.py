import math

import random
import numpy as np
import random
import numpy as np
import math
import shutil


class GeometryCalculator:
    '''
    Class that encapsulates some special geometrical calculations
    '''
    def __init__(self):
        self.earthRadius = 6378.14 *10**3

    def getAngleBetweenVectors(self, vector1, vector2):
        
        angleRadians = np.abs(np.arccos(np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))))
        angleDegrees = angleRadians*(180/np.pi)
        return angleDegrees

    def pointInsideBox(self, point, xMin, xMax, yMin, yMax):
        xCoord = point[0]
        yCoord = point[1]

        if(xCoord >= xMin and xCoord <= xMax and yCoord >= yMin and yCoord <= yMax):
            return True
        else:
            return False
        

    
    def getDistanceBetweenCoordinates(self, lat1, lon1, lat2, lon2):

        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        
        # Radius of Earth in meters
        r = 6371000
        
        # Calculate the result
        return c * r
    
    def getBearing(self, latitude1, longitude1, latitude2, longitude2):
        lat1 = math.radians(latitude1)
        lat2 = math.radians(latitude2)

        diffLong = math.radians(longitude2 - longitude1)

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)* math.cos(lat2)*math.cos(diffLong))

        initialBearing = math.atan2(x, y)


        initialBearing = math.degrees(initialBearing)
        compassBearing = (initialBearing + 360) % 360

        return compassBearing
    
    def getPointProjection(self, vector1, vector2, point):
        A = np.array(vector1)
        B = np.array(vector2)
        P = np.array(point)

        # Vector AB
        vectorAB = B - A
        # Vector AP
        vectorAP = P - A

        # Project vector AP onto vector AB
        dotProductABAB = np.dot(vectorAB, vectorAB)
        if dotProductABAB == 0:
            return A  # A and B are the same point
        projectionScalar = np.dot(vectorAP, vectorAB) / dotProductABAB

        # Clamp the projection scalar to ensure the projected point lies on the segment
        projectionScalarClamped = np.clip(projectionScalar, 0, 1)

        # Calculate the projection point using the clamped scalar
        projectionPoint = A + projectionScalarClamped * vectorAB

        return projectionPoint