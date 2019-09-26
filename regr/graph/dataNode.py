if __package__ is None or __package__ == '':
    from graph import Graph
else:
    from .graph import Graph
    
# Class representing single data instance
class DataNode:
   
    def __init__(self, instance = None, ontologyNode = None, childInstanceNodes = None):
        self.instance = instance                     # The data instance (e.g. paragraph number, sentence number, token number, etc.)
        self.ontologyNode = ontologyNode             # Reference to the ontology graph node which this instance of 
        self.childInstanceNodes = childInstanceNodes # List child data nodes this instance was segmented into
        self.calculatedTypesOfPredictions = dict()   # Dictionary with types of calculated predictions (for learned model, from constrain solver, etc.) results for elements of the instance
        
    def getInstance(self):
        return self.instance
    
    def getOntologyNode(self):
        return self.ontologyNode
    
    def getChildInstanceNodes(self):
        return self.childInstanceNodes
    
    # Get set of types of calculated prediction stored in the graph
    def getCalculatedTypesOfPredictions(self):
        return self.calculatedTypesOfPredictions

    # Set the prediction result data of the given type for the given concept
    def setPredictionResultForConcept(self, typeOfPrediction, concept, prediction=None):
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            self.calculatedTypesOfPredictions[typeOfPrediction] = dict()
            
        if "Concept" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"] = dict()
        
        self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][concept.name()] = prediction
   
    # Set the prediction result data of the given type for the given relation, positionInRelation indicates where is the current instance
    def setPredictionResultForRelation(self, typeOfPrediction, positionInRelation, relation, *instances, prediction=None):
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            self.calculatedTypesOfPredictions[typeOfPrediction] = dict()
            
        if "Relation" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"] = dict()
             
        if relation.name() not in self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"]:
            self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][relation.name()] = dict()
            
        instanceRelationDict = self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][relation.name()]
               
        updatedInstances = instances[0 : positionInRelation - 1] + (self.instance,) + instances[positionInRelation-1: ]
        
        for relatedInstanceIndex in range(len(updatedInstances) - 1):    
            currentInstance = updatedInstances[relatedInstanceIndex] 
            
            if currentInstance not in instanceRelationDict:
                instanceRelationDict[currentInstance] = dict()
            
            instanceRelationDict = instanceRelationDict[currentInstance]
        
        instanceRelationDict[updatedInstances[len(updatedInstances) - 1]] = prediction
   
    # Get the prediction result data of the given type for the given concept
    def getPredictionResultForConcept(self, typeOfPrediction, concept):
        myReturn = None
        
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            pass
        elif "Concept" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            pass
        elif concept.name() not in self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"]:
            pass
        else:
            myReturn = self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][concept.name()]
                                                                                         
        if (myReturn is None) and (self.learnedModel is not None):
            myReturn = concept.getPredictionFor(self.instance)
            
            if myReturn != None:
                self.setPredictionResultForConcept(typeOfPrediction, concept.name(), prediction = myReturn)
            
        return myReturn
            
    # Get the prediction result data of the given type for the given concept or relation (then *tokens is not empty)
    def getPredictionResultForRelation(self, typeOfPrediction, positionInRelation, relation, *instances):
        myReturn = None

        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            pass         
        elif "Relation" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            pass
        else:
            instanceRelationDict = self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][relation.name()]
            
            updatedInstances = instances[0 : positionInRelation - 1] + (self.instance,) + instances[positionInRelation-1: ]

            for relatedInstanceIndex in range(len(updatedInstances) - 1):    
                currentInstance = updatedInstances[relatedInstanceIndex] 
                
                if currentInstance not in instanceRelationDict:
                    break
                
                instanceRelationDict = instanceRelationDict[currentInstance]
                
                if relatedInstanceIndex == len(updatedInstances) - 2:
                    myReturn = instanceRelationDict[updatedInstances[len(updatedInstances) - 1]]
            
        if (myReturn is None) and (self.learnedModel is not None):
            updatedInstances = instances[0 : positionInRelation - 1] + (self.instance,) + instances[positionInRelation-1: ]
            myReturn = relation.getPredictionFor(updatedInstances)
            
            if myReturn != None:
                self.setPredictionResultForRelation(typeOfPrediction, relation.name(), instances, prediction = myReturn)
            
        return myReturn       