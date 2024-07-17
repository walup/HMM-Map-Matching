from PathMappingUtilities import GeometryCalculator
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import numpy as np

class MapGraph:
    
    def __init__(self):
        self.graph = nx.Graph()
        self.nodeDict = {}
    
    def nodeExists(self, pointId):
        return self.graph.has_node(pointId)
    
    def edgeExists(self, point1Id, point2Id):
        return self.graph.has_edge(point1Id, point2Id)
    
    def addPoint(self, pointId, longitude, latitude):
        if(not self.nodeExists(pointId)):
            self.graph.add_node(pointId, pos = (longitude, latitude))
            self.nodeDict[pointId] = (longitude, latitude)
        else:
            print("Node already added")
    
    def addConnection(self, point1Id, point2Id):
        if(self.nodeExists(point1Id) and self.nodeExists(point2Id) and not self.edgeExists(point1Id, point2Id)):
            coords1 = self.nodeDict[point1Id]
            coords2 = self.nodeDict[point2Id]
            geometryCalculator = GeometryCalculator()
            weight = geometryCalculator.getDistanceBetweenCoordinates(coords1[1], coords1[0], coords2[1], coords2[0])
            self.graph.add_edge(point1Id, point2Id, weight = weight)
    
    def getShortestPath(self, node1, node2):
        dst = nx.shortest_path_length(self.graph, node1, node2, weight = "weight")
        return dst
    

    def drawMeasurementsAndPath(self, measurements, path, edgeSize = 2, nodeSize = 1, edgeColor = "#ff884d", nodeColor = "#085eff", pathEdgeColor = "#00e61f", width = 5, height = 5):
        plt.figure(figsize = (width, height))
        edges = list(self.graph.edges)

        #First draw all edges
        for i in range(0,len(edges)):
            edge = edges[i]
            point1 = self.nodeDict[edge[0]]
            point2 = self.nodeDict[edge[1]]
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], marker = "none", color = edgeColor, linewidth = edgeSize)
        

        #Draw the path
        for i in range(0,len(path)):
            edge = edges[int(path[i])]
            point1 = self.nodeDict[edge[0]]
            point2 = self.nodeDict[edge[1]]
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], marker = "none", color = pathEdgeColor, linewidth = edgeSize)
        

        #Draw the measurements
        print(measurements)
        plt.plot(measurements[:,0], measurements[:,1], marker = "o", linestyle = "none", color = nodeColor, markersize = nodeSize)
        

    
    def estimateDistance(self, projectedPoint1, projectedPoint2, edge1, edge2):
        '''
        Projected point1 is the start point which has to belong to the segment delimited by edge1.
        Projected point2 is the end point which has to belong to the segment delimited by edge2
        edge1 is the start edge
        edge2 is the end edge
        '''

        candidateDistances = []
        geometryCalculator = GeometryCalculator()

        #There are many possibilities and we need to consider all of them. 
        startEdgePoint1 = self.nodeDict[edge1[0]]
        startEdgePoint2 = self.nodeDict[edge1[1]]
        endEdgePoint1 = self.nodeDict[edge2[0]]
        endEdgePoint2 = self.nodeDict[edge2[1]]


        dst1 = geometryCalculator.getDistanceBetweenCoordinates(projectedPoint1[1], projectedPoint1[0], startEdgePoint1[1], startEdgePoint1[0])
        dst2 = geometryCalculator.getDistanceBetweenCoordinates(projectedPoint1[1], projectedPoint1[0], startEdgePoint2[1], startEdgePoint2[0])
        dst3 = geometryCalculator.getDistanceBetweenCoordinates(projectedPoint2[1], projectedPoint2[0], endEdgePoint1[1], endEdgePoint1[0])
        dst4 = geometryCalculator.getDistanceBetweenCoordinates(projectedPoint2[1], projectedPoint2[0], endEdgePoint2[1], endEdgePoint2[0])


        #First case
        distance = 0
        try:

            distance = dst1 + self.getShortestPath(edge1[0], edge2[1])+dst4
        except:
            distance = float('inf')
        
        candidateDistances.append(distance)
    
        try:
            distance = dst1 +self.getShortestPath(edge1[0], edge2[0])+dst3
        except:
            distance = float('inf')
        
        candidateDistances.append(distance)


        try:
            distance = dst2 +self.getShortestPath(edge1[1], edge2[1])+dst4
        except:
            distance = float('inf')
        
        candidateDistances.append(distance)

        try:
            distance = dst2 +self.getShortestPath(edge1[1], edge2[0])+dst3
        except:
            distance = float('inf')
        
        candidateDistances.append(distance)
    
        finalDistance = np.min(candidateDistances)

        return finalDistance


    def drawGraph(self, ax, edgeSize = 2, nodeSize = 2):
        pos=nx.get_node_attributes(self.graph,'pos')
        nx.draw(self.graph, ax = ax, node_size = nodeSize, width = edgeSize, pos = pos)


class ViterbiSolver:

    def viterbiDiscrete(self, observations, states, startProb, transProbs, emitProbs):
        """
        Apply the Viterbi algorithm for a time-inhomogeneous HMM with discrete observations.

        observations: list of observed values (discrete)
        states: list of states
        startProb: initial state probabilities
        transProbs: list of transition matrices, one for each time step
        emitProbs: emission probabilities matrix
        """
        T = len(observations)
        N = len(states)

        # Initialize the delta and psi matrices
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        # Initialize the first column of delta and psi
        for i in range(N):
            delta[0, i] = startProb[i] * emitProbs[i, observations[0]]
            psi[0, i] = 0

        # Recursion step
        for t in range(1, T):
            for j in range(N):
                maxVal = -1
                maxState = -1
                for i in range(N):
                    val = delta[t-1][i] * transProbs[i,j,t-1]
                    if val > maxVal:
                        maxVal = val
                        maxState = i
                delta[t, j] = maxVal * emitProbs[j, observations[t]]
                psi[t, j] = maxState

        # Termination
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1, :])
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1][path[t+1]]

        return path, delta, psi

    




class HMMMapMatcher:
    
    def __init__(self, mapGraph, sigma, beta):
        self.mapGraph = mapGraph
        self.sigma = sigma
        self.beta = beta
    
    def computePath(self, measurements, verbose = True):
        '''
        Measurements is a table where each row is a (longitude, latitude) pair
        '''
        
        self.measurements = measurements
        
        if(verbose):
            print("Cleaning measurements")
        
        cleanedMeasurements = self.cleanMeasurements(measurements)
        
        if(verbose):
            print("Obtaining measurement probabilities")
            
        result = self.correctBreakConditionForFunction(self.getMeasurementProbabilities, cleanedMeasurements)
        #print(result)
        if(result is None):
            print("Unsolvable break")
        
        if(verbose):
            print("Computing transition probabilities")
        
        transitionProbabilities = self.getCompleteTransitionProbabilityMatrix(cleanedMeasurements)

        if(verbose):
            print("Computing initial probabilities")
        

        initialProbabilities = self.getInitialProbabilities(cleanedMeasurements)

        viterbiSolver = ViterbiSolver()
        edges = list(self.mapGraph.graph.edges)
        nEdges = len(edges)

        x, delta, psi = viterbiSolver.viterbiDiscrete([i for i in range(0,np.size(cleanedMeasurements,0))],[i for i in range(0,nEdges)], initialProbabilities, transitionProbabilities, result)


        
        return x

        
    def correctBreakConditionForFunction(self, func, measureArray):
        condition = False
        result = None
        while(condition == False and len(measureArray) > 0):
            condition, result = func(measureArray)
            if(condition == False):
                print(measureArray)
                print(result)
                np.delete(measureArray,result, 1)
        
        if(len(measureArray) == 0):
            return None
        else:
            return result
    
    def drawMeasurementsAndPath(self, measurements, path, nodeSize = 2, edgeSize = 2):
        self.mapGraph.drawMeasurementsAndPath(measurements, path, nodeSize = nodeSize, edgeSize = edgeSize)
    
    def cleanMeasurements(self, measurements):
        '''
        Successive measurements need to have a distance bigger than 2*sigma
        '''
        nMeasurements = len(measurements)
        filteredMeasurements = []
        geometryCalculator = GeometryCalculator()
        for i in range(0,nMeasurements-1):
            point1 = measurements[i,:]
            point2 = measurements[i+1,:]
            
            dst = geometryCalculator.getDistanceBetweenCoordinates(point1[1],point1[0],point2[1],point2[0])
            if(dst >= 2*self.sigma):
                filteredMeasurements.append(point1)
            
            if(i == nMeasurements - 2):
                filteredMeasurements.append(measurements[nMeasurements - 1,:])
        
        return np.array(filteredMeasurements)
            
    
    def getMeasurementProbabilities(self, measurements):
        '''
        Returns:
        1. A boolean value indicating whether or not there is a break condition
        2. If (1) is false a list of the indices were no nearby streets were found
        3. If (1) is true a table p(ri, zj) of the probability of obtaining measurement j 
        given edge (road) i
        '''
        edges = list(self.mapGraph.graph.edges)
        nEdges = len(edges)
        nMeasurements = np.size(measurements, 0)
        probs = np.zeros((nEdges, nMeasurements))
        for i in range(0,nEdges):
            edge = edges[i]
            for j in range(0,nMeasurements):
                #Get projection into edge j
                measurement = measurements[j,:]
                probs[i,j] = self.getMeasurementProbability(edge, measurement)
        
        #Break condition
        #See if any column is full of zeroes
        zeroCols = self.getIndicesToDelete(probs)
        if(len(zeroCols) > 0):
            return False, zeroCols
        
        else:
            return True, probs
    
    def getTransitionProbability(self, edge1, edge2, measurement1, measurement2, threshold = 100):
        geometryCalculator = GeometryCalculator()
        point1 = self.mapGraph.nodeDict[edge1[0]]
        point2 = self.mapGraph.nodeDict[edge1[1]]
        projectionPoint1 = geometryCalculator.getPointProjection(point1, point2, measurement1)
        point1 = self.mapGraph.nodeDict[edge2[0]]
        point2 = self.mapGraph.nodeDict[edge2[1]]
        projectionPoint2 = geometryCalculator.getPointProjection(point1, point2, measurement2)

        distance1 = geometryCalculator.getDistanceBetweenCoordinates(measurement1[1], measurement1[0], measurement2[1], measurement2[0])
        distance2 = self.mapGraph.estimateDistance(projectionPoint1, projectionPoint2, edge1, edge2)
        d = np.abs(distance1 - distance2)
        if(d > threshold):
            return 0
        
        else:
            return (1/self.beta)*np.exp(-d/self.beta)

    def getMeasurementProbability(self, edge, measurement, threshold = 100):
        point1 = self.mapGraph.nodeDict[edge[0]]
        point2 = self.mapGraph.nodeDict[edge[1]]
        geometryCalculator = GeometryCalculator()
        projectionPoint = geometryCalculator.getPointProjection(point1, point2, measurement)
        dst = geometryCalculator.getDistanceBetweenCoordinates(measurement[1], measurement[0], projectionPoint[1], projectionPoint[0])
        #print(f'edge {edge[0]} -> {edge[1]} measurement {measurement} = {dst}')
        if(dst <= threshold):
            return (1/(np.sqrt(2*np.pi)*self.sigma))*np.exp(-0.5*(dst/self.sigma)**2)
        else:
            return 0
            

    def getInitialProbabilities(self, measurements):
        #initial Probabilities are obtained with the first measurement
        firstMeasurement = measurements[0,:]
        edges = list(self.mapGraph.graph.edges)
        nEdges = len(edges)
        initialProbabilities = np.zeros(nEdges)
        for i in range(0,nEdges):
            edge = edges[i]
            probability = self.getMeasurementProbability(edge, firstMeasurement)
            initialProbabilities[i] = probability
        
        return initialProbabilities
    




    def getTransitionProbabilities(self, index, measurements, distanceThreshold = 50):
        '''
        Returns the transition probability matrix at time index. 
        The index is in the range [0,N-1] where N is the number of measurements
        '''


        edges = list(self.mapGraph.graph.edges)
        nEdges = len(edges)
        transitionProbabilities = np.zeros((nEdges, nEdges))
        measurement = measurements[index,:]
        nextMeasurement = measurements[index+1, :]

        for i in range(0,nEdges):
            edge1 = edges[i]
            for j in range(i,nEdges):
                edge2 = edges[j]
                probability = self.getTransitionProbability(edge1, edge2, measurement, nextMeasurement, threshold = distanceThreshold)
                transitionProbabilities[i,j] = probability
                transitionProbabilities[j,i] = probability
        

        return transitionProbabilities
    

    def getCompleteTransitionProbabilityMatrix(self, measurements):
        nMeasurements = np.size(measurements, 0)
        edges = list(self.mapGraph.graph.edges)
        nEdges = len(edges)
        completeProbabilities = np.zeros((nEdges, nEdges, nMeasurements-1))
        for i in range(0,nMeasurements-1):
            completeProbabilities[:,:,i] = self.getTransitionProbabilities(i, measurements)
        
        return completeProbabilities

    
    def getIndicesToDelete(self, matrix):
        a = [i for i, col in enumerate(np.transpose(matrix)) if not any(col)]
        n = np.size(matrix, 1)
        newArr = []
        if(len(a) > 0):
            for i in range(0,len(a)):
                newArr.append(a[i])
                if(a[i]+1 < n):
                    newArr.append(a[i]+1)
    
        return newArr
                
                
                
        
        