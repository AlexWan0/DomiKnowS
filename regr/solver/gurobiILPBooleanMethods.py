import logging
from itertools import permutations

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.ilpConfig import ilpConfig 

from gurobipy import *

class gurobiILPBooleanProcessor(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
                
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']

        self.constrainCaches = {}

    def resetCaches(self):
         self.constrainCaches = {}

    def __addToConstrainCaches(self, lmName, onlyConstrains, var, cachedValue):
        if lmName in self.constrainCaches:
                if onlyConstrains in self.constrainCaches[lmName]:
                    self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
                else:
                    self.constrainCaches[lmName][onlyConstrains] = {}
                    self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
        else:
            self.constrainCaches[lmName] = {}
            self.constrainCaches[lmName][onlyConstrains] = {}

            self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
            
    def __isInConstrainCaches(self, lmName, onlyConstrains, var):
        if lmName in self.constrainCaches:
            if onlyConstrains in self.constrainCaches[lmName]:
                for currentVarPermutation in permutations(var):
                    if currentVarPermutation in self.constrainCaches[lmName][onlyConstrains]:
                        #if self.ifLog: self.myLogger.debug("%s already created constrain for this variables %s - does nothing"%(lmName, [x.VarName for x in var]))
                        return (True, self.constrainCaches[lmName][onlyConstrains][currentVarPermutation])
                    
        return (False, None)
                
    def notVar(self, m, var, onlyConstrains = False):
        methodName = "notVar"
        logicMethodName = "NOT"
        
        varName = var
        if isinstance(var, Var):
            varName = var.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,varName))

        if isinstance(var, Var):
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var,))
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constrain already created - doing nothing"(logicMethodName))
                return cacheResult[1]
            
        # If this is a standalone or head method - do not create ILP variable for OR        
        if onlyConstrains:
            if not isinstance(var, Var):
                self.myLogger.warn("%s has set value: %s - do nothing"%(logicMethodName,var1Name))
                return 
            
            m.addConstr(1 - var >= 1)
            if self.ifLog: self.myLogger.debug("%s created constrain only: not %s > %i"%(logicMethodName,var.VarName,1))

            #self.__addToConstrainCaches(methodName, onlyConstrains, (var, ), None)
            return
        
        if not isinstance(var, Var):
            if var == 0:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,0))
                return 0
            
        varNOT = m.addVar(vtype=GRB.BINARY, name="not_%s"%(var.VarName))
        
        m.addConstr(1 - var == varNOT)
        if self.ifLog: self.myLogger.debug("%s created constrain: %i - %s == %s "%(logicMethodName,VarName,varNOT.VarName))

        m.update()
        if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varNOT.VarName))
            
        self.__addToConstrainCaches(methodName, onlyConstrains, (var,), varNOT)          
        return varNOT
    
    def and2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "and2Var"
        logicMethodName = "AND"
        
        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if isinstance(var1, Var):
            var1Name = var1.VarName
        if  isinstance(var2, Var):
            var2Name = var2.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1Name,var2Name))
        
        # Check caches
        if isinstance(var1, Var) and isinstance(var2, Var):
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var1, var2))
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constrain already created - doing nothing"%(logicMethodName))
                return cacheResult[1]
        
        # If this is a standalone or head method - do not create ILP variable for OR        
        if onlyConstrains:
            if not isinstance(var1, Var):
                if var1 == 1:
                    m.addConstr(var2 >= 1)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= 1"%(logicMethodName,var2Name))
                    return
                else:
                    self.myLogger.error("%s always False: %s is 0"%(logicMethodName,var1Name))
                    return 
            if  not isinstance(var2, Var):
                if var2 == 1:
                    m.addConstr(var1 >= 1)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= 1"%(logicMethodName,var1Name))
                    return
                else:
                    self.myLogger.error("%s always False: %s is 0"%(logicMethodName,var2Name))
                    return 
            else: # Both variables are ILP variables
                m.addConstr(var1 + var2 >= 2) 
                if self.ifLog: self.myLogger.debug("% created constrain only: and %s %s >= 2"%(logicMethodName,var1.VarName,var2.Name))
    
                #self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), None)
                return
        
        if not isinstance(var1, Var):
            if var1 == 0:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,0))
                return 0
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var2Name))
                return var2
        if  not isinstance(var2, Var):
            if var2 == 0:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,0))
                return 0
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var2Name))
                return var1
        else:
            varAND = m.addVar(vtype=GRB.BINARY, name="and_%s_%s"%(var1Name, var2Name))
    
            m.addConstr(varAND - var1 <= 0) # varAND <= var1
            if self.ifLog: self.myLogger.debug("%s created constrain: %s  - %s <= 0 "%(logicMethodName,varAND.VarName,var1Name))

            m.addConstr(varAND - var2 <= 0) # varAND <= var2
            if self.ifLog: self.myLogger.debug("%s created constrain: %s  - %s <= 0 "%(logicMethodName,varAND.VarName,var2Name))

            m.addConstr(var1 + var2 - varAND <= 1) # var1 + var2 <= varAND + 2 - 1
            if self.ifLog: self.myLogger.debug("%s created constrain: %s + %s  - %s <= 1 "%(logicMethodName,VarName,var1Name,varAND.VarName,var2Name))

            # Update cache
            self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), varAND) 
            
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varAND.VarName))
            m.update()
            return varAND
    
    def andVar(self, m, *var, onlyConstrains = False):
        methodName = "andVar"
        logicMethodName = "AND"
        #if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,var))
        
        N = len(var)

        if N <= 1:
            return None
                
        if N == 2:
            return self.and2Var(m, var[0], var[1], onlyConstrains)
        
        cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var1, var2))
        if cacheResult[0]:
            if self.ifLog: self.myLogger.debug("%s constrain already created - doing nothing"(logicMethodName))
            return cacheResult[1]
        
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            if self.ifLog: self.myLogger.debug("% created constrain only: and %s > 1"%(logicMethodName,varSumLinExpr))

            #self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), None)
            m.addConstr(varSumLinExpr >= N)
            return
            
        varAND = m.addVar(vtype=GRB.BINARY)
        for currentVar in var:
            m.addConstr(varAND - currentVar <= 0) # varAND <= currentVar

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
        
        m.addConstr(varSumLinExpr - varAND <= N - 1) # varSumLinExpr <= varAND + N - 1
    
        m.update()
        if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varAND.VarName))
            
        self.__addToConstrainCaches(methodName, onlyConstrains, var, varAND) 
        return varAND
    
    def or2Var(self, m, var1, var2, onlyConstrains = False, limit = 1):
        methodName = "or2Var"
        logicMethodName = "OR"
        
        if limit > 1:
            if self.ifLog: self.myLogger.debug("%s called with limit: %i"%(logicMethodName,limit))

        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if isinstance(var1, Var):
            var1Name = var1.VarName
        if  isinstance(var2, Var):
            var2Name = var2.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1Name,var2Name))

        # Check caches
        if isinstance(var1, Var) and isinstance(var2, Var):
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var1, var2))
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constrain already created"%(logicMethodName))
                if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))
                return cacheResult[1]
        
        # If this is a standalone or head method - do not create ILP variable for OR
        if onlyConstrains:
            if not isinstance(var1, Var):
                if var1 == 0:
                    m.addConstr(var2 >= 1)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= %i"%(logicMethodName,var2Name,1))
                    return
                else:
                    if self.ifLog: self.myLogger.debug("%s created no constrain first variable is already %i"%(logicMethodName,1))
                    return
            if  not isinstance(var2, Var):
                if var2 == 0:
                    m.addConstr(var1 >= 1)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= %i"%(logicMethodName,var1Name,1))
                    return
                else:
                    if self.ifLog: self.myLogger.debug("%s created no constrain second variable is already 1"%(logicMethodName))
                    return
            else:
                m.addConstr(var1 + var2 >= limit) 
                if self.ifLog: self.myLogger.debug("%s created constrain only: %s  + %s >= %i"%(logicMethodName,var1Name,var2Name,limit))

                #self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), None)
                return
        
        if not isinstance(var1, Var): 
            if var1 == 1:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var2Name))
                return var2
        if  not isinstance(var2, Var):
            if var2 == 1:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var1Name))
                return var1
        else: # Both variables are ILP variables
            varOR=m.addVar(vtype=GRB.BINARY, name="or_%s_%s"%(var1Name, var2Name))
            m.update()

            m.addConstr(var1 - varOR <= 0) # var1 <= varOR
            if self.ifLog: self.myLogger.debug("%s created constrain: %s  - %s <= %i"%(logicMethodName,var1Name,var2Name,0))

            m.addConstr(var2 - varOR <= 0) # var2 <= varOR
            if self.ifLog: self.myLogger.debug("%s created constrain: %s  - %s <= %i"%(logicMethodName,var2Name,var2Name,0))

            m.addConstr(var1 + var2 - varOR >= limit-1) # var1 + var2 >= varOR
            if self.ifLog: self.myLogger.debug("%s created constrain: %s + %s - %s >= %i"%(logicMethodName,var1Name,var2Name,varOR.VarName,limit-1))

            # Update cache
            self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), varOR) 
            
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varOR.VarName))
            m.update()
            return varOR
    
    def orVar(self, m, *var, onlyConstrains = False, limit = 1):
        methodName = "orVar"
        logicMethodName = "OR"
        
        if limit > 1:
            if self.ifLog: self.myLogger.debug("%s called with limit: %i"%(logicMethodName,limit))

        # Get names of variables - some of them can be numbers
        allVars = 0
        varsNames = []
        for currentVar in var:
            if isinstance(currentVar, Var):
                varsNames.append(currentVar.VarName)
                allVars = allVars + 1
            else:
                varsNames.append(currentVar)
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, varsNames))
        
        # Check number of variables
        N = len(var)
        
        if N <= 1:
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,'None'))
            return None
        
        if N == 2: return self.or2Var(m, var[0], var[1], onlyConstrains)
        
        # If all are variables then check cache
        if N - allVars == 0:
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, var)
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constrain already created - doing nothing"%(logicMethodName))
                if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))
                return cacheResult[1]
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                if isinstance(currentVar, Var):
                    varSumLinExpr.addTerms(1.0, currentVar)
                elif currentVar == 1:
                    if self.ifLog: self.myLogger.debug("%s created no constrain variable is already 1"%(logicMethodName))
                    return
                else: 
                    pass
        
            m.addConstr(varSumLinExpr >= limit)
            
            varSumLinExprStr = str(varSumLinExpr)
            if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= %i"%(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],limit))
            
            if N - allVars == 0: self.__addToConstrainCaches(methodName, onlyConstrains, var, None)           
            return
        
        # ------- If creating variables representing value of OR build of provided variables
        
        # If only single variable
        if allVars == 1:
            for currentVar in var:
                if isinstance(currentVar, Var):
                    if self.ifLog: self.myLogger.debug("%s has only single variable: %s, it is returned"%(logicMethodName,currentVar.VarName))
                    return currentVar
                
        # Build new variables name and add it to model
        orVarName = ""
        for currentVar in var:
            orVarName = orVarName + "or"
            if isinstance(currentVar, Var):
                orVarName += "_%s_" % (currentVar.VarName)
            else:
                #orVarName += "_%s_" % (currentVar)
                
                if currentVar == 1:
                    if self.ifLog: self.myLogger.debug("%s created no constrain variable is already 1"%(logicMethodName))
                    return 1

        varOR = m.addVar(vtype=GRB.BINARY, name=orVarName)

        # Build constrains
        for currentVar in var:
            if isinstance(currentVar, Var):
                m.addConstr(currentVar - varOR <= 0) # currentVar <= varOR
                if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s <= %i"%(logicMethodName,currentVar.VarName,orVarName,0))
            else:
                pass # Only 0 possible - has no effect on Or value

        varSumLinExpr = LinExpr()
        for currentVar in var:
            if isinstance(currentVar, Var):
                varSumLinExpr.addTerms(1.0, currentVar)
            else:
                pass
            
        m.addConstr(varSumLinExpr - varOR >= limit-1) # varSumLinExpr >= varOR
        varSumLinExprStr = str(varSumLinExpr)
        if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s >= %i"%(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],orVarName,limit-1))

        # Cache
        if N - allVars == 0: self.__addToConstrainCaches(methodName, onlyConstrains, var, varOR)

        # Return new variable
        m.update()
             
        if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varOR.VarName))
        return varOR
    
    def nand2Var(self, m, var1, var2, onlyConstrains = False):
        #if self.ifLog: self.myLogger.debug("NAND called with : %s"%(var1,var2))

        if onlyConstrains:
            m.addConstr(var1 + var2 <= 1)
            return
        
        varNAND = m.addVar(vtype=GRB.BINARY, name="nand_%s_%s"%(var1, var2))
            
        m.addConstr(self.notVar(m, varNAND) <= var1)
        m.addConstr(self.notVar(m, varNAND) <= var2)
        
        m.addConstr(var1 + var2 <= self.notVar(m, varNAND) + 2 - 1)
        
        return varNAND
    
    def nandVar(self, m, *var, onlyConstrains = False):
        #if self.ifLog: self.myLogger.debug("NAND called with : %s"%(var,))
        
        cacheResult = self.__isInConstrainCaches('nandVar', onlyConstrains, var)
        if cacheResult[0]:
            if self.ifLog: self.myLogger.debug("NAND constrain already created - doing nothing")
            return cacheResult[1]
            
        N = len(var)
        
        if N <= 1:
            if self.ifLog: self.myLogger.debug("NAND returns : %s"%('None'))
            return None
        
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(varSumLinExpr <= N - 1)
                        
            varSumLinExprStr = str(varSumLinExpr)
            if self.ifLog: self.myLogger.debug("NAND created constrain only: %s <= %i"%(varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')], N-1))
            
            self.__addToConstrainCaches('nandVar', onlyConstrains, var, None)
      
            return
        
        nandVarName = "nand"
        for currentVar in var:
            nandVarName += "_%s"%(currentVar)
            
        varNAND = m.addVar(vtype=GRB.BINARY, name=nandVarName)
        for currentVar in var:
            m.addConstr(self.notVar(m, varNAND) <= currentVar)

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
    
        m.addConstr(varSumLinExpr <= self.notVar(m, varNAND) + N - 1)
    
        self.__addToConstrainCaches('nandVar', onlyConstrains, var, varNAND)

        return varNAND
    
    def nor2Var(self, m, var1, var2, onlyConstrains = False):
        if onlyConstrains:
            m.addConstr(var1 + var2 <= 0)
            return
        
        varNOR = m.addVar(vtype=GRB.BINARY, name="nor_%s_%s"%(var1, var2))
            
        m.addConstr(var1 <= self.notVar(m, varNOR))
        m.addConstr(var2 <= self.notVar(m, varNOR))
            
        m.addConstr(var1 + var2 >= self.notVar(m, varNOR))
        
        return varNOR
    
    def norVar(self, m, *var, onlyConstrains = False):
        N = len(var)
        
        if N <= 1:
            return None
        
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(varSumLinExpr <= 0)
            return
        
        _norVarName = "nor"
        for currentVar in var:
            _norVarName += "_%s"%(currentVar)
           
        varNOR = m.addVar(vtype=GRB.BINARY, name=norVarName)
        for currentVar in var:
            m.addConstr(currentVar <= self.notVar(m, varNOR))
        
        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
            
            m.addConstr(varSumLinExpr >= self.notVar(m, varNOR))
    
        return varNOR
    
    def xorVar(self, m, var1, var2, onlyConstrains = False):
        #if self.ifLog: self.myLogger.debug("XOR called with : %s"%(var1,var2))
        
        cacheResult = self.__isInConstrainCaches('xorVar', onlyConstrains, (var1, var2))
        if cacheResult[0]:
            return cacheResult[1]

        if onlyConstrains:
            m.addConstr(var1 + var2 <= 1)
            m.addConstr(var1 + var2 >= 1)
            if self.ifLog: self.myLogger.debug("IF created constrain only: %s <= %s"%(var1.VarName, var2.VarName))

            self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), None)
            return
        
        varXOR = m.addVar(vtype=GRB.BINARY, name="xor_%s_%s"%(var1, var2))
            
        m.addConstr(var1 + var2 + varXOR <= 2)
        m.addConstr(-var1 - var2 + varXOR <= 0)
        m.addConstr(var1 - var2 + varXOR >= 0)
        m.addConstr(-var1 + var2 + varXOR >= 0)
            
        return varXOR
    
    def ifVar(self, m, var1, var2, onlyConstrains = False):
        #if self.ifLog: self.myLogger.debug("IF called with : %s"%(var1,var2))

        if (not var1) or (not var2):
            return
    
        cacheResult = self.__isInConstrainCaches('ifVar', onlyConstrains, (var1, var2))
        if cacheResult[0]:
            if self.ifLog: self.myLogger.debug("IF constrain already created - doing nothing")
            return cacheResult[1]
            
        if onlyConstrains:
            m.addConstr(var1 <= var2)
            if self.ifLog: self.myLogger.debug("IF created constrain only: %s <= %s"%(var1.VarName, var2.VarName))

            self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), None)
            return
        
        varIF = m.addVar(vtype=GRB.BINARY, name="if_%s_then_%s"%(var1.VarName, var2.VarName))
            
        m.addConstr(1 - var1 <= varIF)
        m.addConstr(var2 <= varIF)
        m.addConstr(1 - var1 + var2 >= varIF)
            
        m.update()

        self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), varIF)
        
        if self.ifLog: self.myLogger.debug("IF returns : %s"%(varIF.VarName))

        return varIF
           
    def eqVar(self, m, var1, var2, onlyConstrains = False):
        #if self.ifLog: self.myLogger.debug("EQ called with : %s"%(var1,var2))

        cacheResult = self.__isInConstrainCaches('eqVar', onlyConstrains, (var1, var2))
        if cacheResult[0]:
            if self.ifLog: self.myLogger.debug("EQ constrain already created - doing nothing")
            return cacheResult[1]
        
        if onlyConstrains:
            m.addConstr(var1 >= var2)
            if self.ifLog: self.myLogger.debug("EQ created constrain only: %s => %s"%(var1.VarName, var2.VarName))
            
            m.addConstr(var1 <= var2)
            if self.ifLog: self.myLogger.debug("EQ created constrain only: %s <= %s"%(var1.VarName, var2.VarName))

            self.__addToConstrainCaches('eqVar', onlyConstrains, (var1, var2), None)
            return
        
        varEQ = m.addVar(vtype=GRB.BINARY, name="epq_%s_%s"%(var1, var2))
            
        m.addConstr(var1 + var2 - varEQ <= 1)
        m.addConstr(var1 + var2 + varEQ >= 1)
        m.addConstr(-var1 + var2 + varEQ <= 1)
        m.addConstr(var1 - var2 + varEQ <= 1)
        
        m.update()
             
        self.__addToConstrainCaches('eqVar', onlyConstrains, (var1, var2), varEQ)

        if self.ifLog: self.myLogger.debug("EQ returns : %s"%(varEQ.VarName))
        return varEQ