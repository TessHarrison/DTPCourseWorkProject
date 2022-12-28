#Imports
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
import matplotlib.pyplot as plt
import os.path
from scipy import special
import math

#Classes
class individualParticipentData():
    """
    class containing and managing individual participent data, cos python is painfully object orientated
    this is generally the easiest way to handle stuff like this, also aids for doing manipulations within
    the class and general organisation
    """
    attKeys = ["attr","sinc","intel","fun","amb"]
    nOfKeys = len(attKeys)
    #attrative, sincere, intelligent, fun, ambitious,
    #const used to automate attribute extraction

    #function to spit out a dict automatically, of attributes
    #taken from the data at a specific time and w/ specific context
    #as automated by the attA_B notation used (described in notebook)
    #note: defaults to no time code
    def genAttDict(self,dataEntry, contextCode, timeCode=""):
        rtnDict = dict.fromkeys(self.attKeys) #preallocate dict
        for rootKey in self.attKeys: #loop through the key list
            rtnDict[rootKey] = dataEntry[f"{rootKey}{contextCode}{timeCode}"] #add each attribute w/ said parameters
        return rtnDict

    #constructor, set up to read directly from the data entry in the dataframe 
    #see accompanying notebook for info as to what codes mean
    #this is gonna be just a bunch of orgnaisation
    #unfortunately with the way this is organised its difficult to automate this
    #so this is gonna look kinda gross
    #essentially I could just grab everything, but I don't want it all so we need to specify
    #techically we could grab everything except a black list but again I want to reorganise it so :/
    def __init__(self, dataEntry):
        self.iid = dataEntry["iid"] #primary key
        self.metadata = {
            "gender":dataEntry["gender"],
            "expHappy":dataEntry["exphappy"],
            "expInterest":dataEntry["expnum"],
            "careerIntent":dataEntry["career_c"],
            "outFreq":dataEntry["go_out"],
            "dateFreq":dataEntry["date"],
            "dateGoal":dataEntry["goal"],
            "localIncome":dataEntry["income"],
            "zipcode":dataEntry["zipcode"],
            "ungrdTutition":dataEntry["undergra"],
            "study":dataEntry["field_cd"],
            "age":dataEntry["age"],
            "wave":dataEntry["wave"], #this refers to the group of speed dating they were in
            "interestSports":dataEntry["sports"],
            "interestTVSports":dataEntry["tvsports"],
            "interestExercise":dataEntry["exercise"],
            "interestDining":dataEntry["dining"],
            "interestMusic":dataEntry["music"],
            "interestArt":dataEntry["art"],
            "interestHiking":dataEntry["hiking"],
            "interestGaming":dataEntry["gaming"],
            "interestClubbing":dataEntry["clubbing"],
            "interestReading":dataEntry["reading"],
            "interestTV":dataEntry["tv"],
            "interestTheatre":dataEntry["theater"],
            "interestMovies":dataEntry["movies"],
            "interestConcerts":dataEntry["concerts"],
            "interestMuseums":dataEntry["museums"],
            "interestShopping":dataEntry["shopping"],
            "interestYoga":dataEntry["yoga"]
        }

        #grab the attributes people assign to themselves
        self.selfAtt = {
            "before":self.genAttDict(dataEntry,"3","_1"),
            "during":self.genAttDict(dataEntry,"3","_s"),
            "afterEvt":self.genAttDict(dataEntry,"3","_2"),
            "afterDat":self.genAttDict(dataEntry,"3","_3")
        }

        #preinitalise the attributes others assign
        self.extAtt = []

    #equality overide, allows usage of == operator to compare objects
    #as iid is the primary key we simply check on that, and the type of the obj
    #may be redudnant but useful to have 
    def __eq__(self, other):
        if type(other) is individualParticipentData: #is this of this class?
            if other.iid == self.iid: #is it the same participent
                return True

        return False #its a different class and/or a different participent

    def isSameIid(self, testIid):
        """
        function to test if an iid (int) is equal to this obj's (ie. same person)
        returns true/false
        """
        return self.iid == testIid

    def appendExtAtt(self, dataEntry):
        """
        function to append external perceptions of a persons attributes to the data obj
        """
        self.extAtt.append(self.genAttDict(dataEntry,"_o"))

    def generateRankedDict(self, inputDict):
        """
        function to generate a dict with each key converted to a rank value, note identical values
        are averaged
        """
        #first we're gonna sort the dict using NlogN alogrithms to reduce complexity (the items here is low, so it won't
        # add too much speed but it still should be faster)
        sortedSubDicts = [] # a list of dicts broken in to key and val, to ease comprehension 
        for key in sorted(inputDict,key=inputDict.get): #itterates through keys in order from lowest to highest
            sortedSubDicts.append(
                {"key":key,"val":inputDict[key]} #yes this looks a lot like a dict, but this gives easier index handling
            )
        
        #now we can go through and reletively linearly assign ranks as we know all values only get bigger or stay the same
        currRank = 0
        rankedDict = {}
        #now what we do need to check for is when there are clumps with the same val
        for startIndx in range(len(sortedSubDicts)):
            currVal = sortedSubDicts[startIndx]["val"]
            currIndx = startIndx
            while currIndx+1 < len(sortedSubDicts): #protection from going out of range
                if sortedSubDicts[currIndx+1]["val"] == currVal: #do we have the same val ie have to mean the rank
                    currIndx += 1
                else: break #leave the loop cos we've found all the same value ones

            #now when we leave this loop we need to average the rank for set of idnetical ones we've found
            if startIndx == currIndx: #shortcutting for when there's just one
                rankedDict[sortedSubDicts[startIndx]["key"]] = currRank
            else: #we've got a bunch to add and average
                ranksToAve = range(currRank, currRank+(currIndx-startIndx)+1) #generates the range of rank values to average +1 to make inclusive
                rankToAdd = sum(ranksToAve)/len(ranksToAve) #generates the average rank
                for addIndx in range(startIndx, currIndx+1): #loop through the indicies
                    rankedDict[sortedSubDicts[addIndx]["key"]] = rankToAdd
                startIndx = currIndx #sets the counter to the end of the ones we just sorted, so the loop doesn't overlap things
        
        #here we should be all good
        return rankedDict


    def calcCorrelation(self,dictA,dictB):
        """
        function to calculate the correlation between two attribute dicts
        this is calculated via a spearman's rank 
        """
        #rank the dicts
        rankedA = self.generateRankedDict(dictA)
        rankedB = self.generateRankedDict(dictB)
        #calc the difs 
        sumOfSqrDiffs = 0
        for key in self.attKeys:
            sumOfSqrDiffs += (rankedA[key] - rankedB[key]) ** 2
        
        val = 1-((6*sumOfSqrDiffs)/(self.nOfKeys*((self.nOfKeys**2)-1)))
        if abs(val) > 1:
            raise ValueError("Spearmans rank out of expected bounds")
        return val

    def calcAverageCorrelations(self,personalTiming):
        """
        function to calculate the average correlation between a persons perceptions of themselves
        vs the external perceptions of them
        personalTiming can be - before,during,afterEvt,afterDat the latter two being after the 
        speeddating event/after time to allow matches to go on date(s)
        """
        #calculate a correlation between the self and other perceptions for each partner
        corrList = []
        for extAttDict in self.extAtt:
            corrList.append(self.calcCorrelation(extAttDict, self.selfAtt[personalTiming]))

        #return the average
        if len(corrList) == 0: #to dodge divide by zero errors if something wasn't added 
            return pd.NA
        return sum(corrList)/len(corrList)

    def calcAllCorrelations(self):
        """
        function to whiz through and calculate aggreement at each time point
        """
        corrDict = {}
        for key in self.selfAtt.keys():
            corrDict["corr_"+key] = self.calcAverageCorrelations(key)
        
        self.perceptionAgreementDict = corrDict #slap it on the object so its easy to handle 
        return corrDict #just in case you want to do anything w/ it 

    def exportToDataLine(self, dataType="dict"):
        """
        Function to export the data in a line that you can slap onto a table or what not
        by default exports as dict, but can be overidden to a pd.series via "series"
        """
        exportDict = {}
        exportDict.update({"iid":self.iid})
        exportDict.update(self.perceptionAgreementDict)
        exportDict.update(self.metadata)

        if dataType == "series": #can export as a pandas series
            return pd.Series(exportDict, index=exportDict.keys())

        return exportDict #default behaviour, note this will not be executed if the former return statement is



#Functions
def importData(path, encoding_errors):
    """
    function to import the data, used as to reduce memory usage as python garbage handling is garbage 
    this will allow me to temporarily hold the data w/ everything so I can immediately strip it down to
    what we care about
    path - url/file path for data
    encoding_errors - encoding_errors parameter as per pandas.read_csv
    """
    return pd.read_csv(path, encoding_errors=encoding_errors)

def isNumber(potentialNumber):
    """
    good ol' error trap to see if something is a number or not
    """
    if str(potentialNumber) == "nan": return False #annoyingly nans/NAs can be interpreted as floats
    try:
        float(potentialNumber) #try to treat as a float
        return True
    except ValueError: #if it blows up its probably not
        return False

def linkAttributes(datingData):
    """
    function which takes the speeddating data, and reformats the data such that it is instead formed of a list of entries per unique person
    containing their metadata, self described attributes and percieved attributes
    """
    #first we're gonna create a list of individual participent data objects
    #this will be initalised with all of the data given by the individual
    #based on the data organisation this should inherently be organised from lowest to highest iid
    #hence we only need to check when the iid changes
    ipdList = []
    for idx, row in datingData.iterrows(): #note this may be faster w/ some sort of list comprehension approach but frankly this sets up nicely to reduce load on the constructor
        if "currIpd" in locals():
            if currIpd.isSameIid(row["iid"]): #we have a repeat person, set up like this to avoid exceptions for it not being initalised
                continue # skip to next row of data
        
        currIpd = individualParticipentData(row) #generate the new obj
        ipdList.append(currIpd) #slap it into the list
        
    #now we're gonna add all the outside perceptions onto those data objects
    #to optimise linking up of these parameters we're gonna first sort the data by partner iid 
    #ie the the iid of the person being described
    #once sorted we'll then go through the list appending each to their respective individual participent data obj
    #this drops complexity from O(N^2) to O(NLog(N)+N)

    datingData.sort_values(by=["pid"], inplace=True) # note: we're sorting here cos we're gonna discard the data when its in our structure
    ipdIdx = 0 #index for ipdList
    emergencyBreakout = datingData["pid"].max(numeric_only=True)+1 #grab the highest iid avalible and add 1 for safety, if we get beyond this somethings gone wrong
                                                                   #this is arbitrary, but is just to prevent infinite loops if something goes wrong
    for idx, row in datingData.iterrows():
        notEvaluated = True
        currPid = row["pid"]
        if isNumber(currPid) == False: #python typing is really horrible to work with so this is one of the easier ways I've found to do this lol
            continue #skip this row if we have a non-numeric value ie something is weird with the pid

        #loop to find the next ipd obj that has the right index (tbh this will probably only ever be the next one)
        while notEvaluated: # bit spicy, but w/ an emergency breakout it should be fine
            if ipdList[ipdIdx].isSameIid(currPid): #are we looking at the right person
                ipdList[ipdIdx].appendExtAtt(row) #add the perceptions to the person
                #note: we can do this as we've just sorted everything such that the order of the data, maps to the order of the idpList
                notEvaluated= False
            else:
                if ipdIdx >= emergencyBreakout:
                    raise ValueError("pid out of range in linkAttributes") #raise an error (will terminate running)
                ipdIdx += 1 

    #so at this point we now have a lovely datastructure w/ all in our objects and everything linked up to 
    #individual people Yay
    return ipdList

def getAgreementLine(obj: individualParticipentData):
    """
    quick handler function to coordiante calculating correlations within the ipd obj and exporting the series so we can then
    slap them all into a massive pd
    """
    obj.calcAllCorrelations() #get the object to calculate how much agreement is going on 
    return obj.exportToDataLine("series") #export the data in a series

def genAgreementData(dataPath):
    """
    function to generate the agreement data of how much people agree w/ others on their self perceptions
    outputs a big panda w/ all the correlation data, and the metadata for the peeps
    """
    ipdList = linkAttributes(importData(dataPath, "ignore")) #get and format all the objects nicely together in a big ol' list with everything linked
    
    #now we just wanna go through and calculate all the data for all the objects, and return them into a new panda
    with ProcessPoolExecutor() as pool:
        exportedSeriesData = pool.map(getAgreementLine, ipdList)
    #exportedSeriesData = [] #for debugging 
    #for obj in ipdList:
    #    exportedSeriesData.append(getAgreementLine(obj))

    return pd.concat(exportedSeriesData,axis=1).T.convert_dtypes() #return the nicely calculated list of data as a dataframe 
    #convert datatypes just optimises the data types for pandas handling (cos python typing is the worst thing known to man - except maybe JS)

def getUniqueFileName(rootName, ext):
    """
    function to test for and generate a unique name to save a file
    """
    currName = rootName+ext #check inital name
    counter = 0
    while os.path.exists(currName): #if its not good, slap an index on it and keep adding
        counter += 1 
        currName = rootName+"_"+str(counter)+ext
    
    return currName
    

def appendTransformedSeries(df, columnKeys, transformationFunc, strAddon="transformed_"):
    """
    function to generate new series via a transformation of specified series within a df
    then slap them onto the end of the df
    """
    for key in columnKeys: #loop through the keys to transfrom
        df[strAddon+key] = df[key].apply(transformationFunc) #add new ones after passing through the func

    return df

def savePlot(idealName):
    """
    function to save a plot w/ handling for not overwriting names etc
    """
    plt.savefig(getUniqueFileName(idealName, ".png"))

def exportDescriptiveStats(idealName,df: pd.DataFrame, columnKeys: list):
    """
    function to export descriptive stats of any columns exported as csv
    """
    outputDict = {}
    for key in columnKeys:
        rowDict = { #calculates all the things to export
            "mean":df[key].mean(),
            "median":df[key].median(),
            "mode":df[key].mode()[0], #as this is exported as a series
            "sd":df[key].std(),
            "sem":df[key].sem(),
            "max":df[key].max(),
            "min":df[key].min()
        }  
        outputDict[key] = rowDict
    outputPd = pd.DataFrame.from_dict(outputDict)
    
    outputPd.to_csv(getUniqueFileName(idealName,".csv")) #save the file

#executable
if __name__ == "__main__":
    #get the data
    print("importing and formatting data")
    personalAttData = genAgreementData("https://query.data.world/s/lxtcrejzkht54b5titjjsymglxwcuw")
    print("plotting scatter matrix")
    #so having looked at the scatter matrix off the raw correlation, you don't see anything coherent
    #as a last ditch, we're gonna try transforming the correlation data as its essentially bounded
    #between 1 and -1, where its harder and harder to get to the extreme, ie. a bit like a arctan(theta)
    #so lets try throwing it through arctan, lets also try tan just in case cos its always funky to work with
    #interestignly, looking at the data, rarely are there negtive correlations in opions ie. rarely do people 
    #also looking more closely to it, the agreement looks fairly squished at the top so lets see how it looks with a 
    #log transform to try and extend it a bit
    corrDataKeys = ["corr_before","corr_during","corr_afterEvt","corr_afterDat"]
    personalAttData = appendTransformedSeries(personalAttData,corrDataKeys,
        lambda x: math.atan(x), "arctan_"
    )
    personalAttData = appendTransformedSeries(personalAttData,corrDataKeys,
        lambda x: math.tan(x), "tan_"
    )
    personalAttData = appendTransformedSeries(personalAttData,corrDataKeys,
        lambda x: math.exp(x), "exp_"
    )
    personalAttData = appendTransformedSeries(personalAttData,corrDataKeys,
        lambda x: math.log(x+1), "log_"
    ) #note: x+1 to avoid log(0) (assuming we're never gonna get a corr of -1 - probability just v low)

    scatMat = pd.plotting.scatter_matrix(personalAttData,figsize=(60,60))
    savePlot("scatterMatrix")

    #well after doing all that, we can't seem to get anything too interesting out of correlating the data
    #lets just export some descriptive stats then as this metadata doesn't appear to have predictive power
    exportDescriptiveStats("descriptiveStats",personalAttData,corrDataKeys)
    personalAttData.to_csv(getUniqueFileName("processedPersonalData",".csv")) #export the processed data if anyone wants it
    



    
    
    