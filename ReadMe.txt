ReadMe
---------------------
Contents of repository:
 - dataExploration.ipynb - JupiterNoteBook containing mininal data exploration and testing
 - Speed Dating Data Key.doc - .doc file from dataset author detailing data collection and data key
 - ArePeopleWhoTheyThinkTheyAre.py - .py script that enables the analysis of individuals opinions of themselves vs the opinions of them from others after a speed dating event

---------------------
Requirements:

Python 3.10.8
(external libraries, most recent versions via pip repository as of 20/12/2022)
pandas
matplotlib
scipy

---------------------
Usage: 

- Run ArePeopleWhoTheyThinkTheyAre.py 
- In the run path 3 files will be created
    - scatterMatrix.png
        A graph showing all combinations of 2d scatter plots
    - descriptiveStats.csv
        descriptive stats on catagories specified
    - processedPersonalData.csv
        table with processed data indicating the correlation thus agreement between self and external perceptions along with avalible personal metadata

 - Notable Functions
    appendTransformedSeries
        can be used to add extra transformed data series to the datatable, will accept any 1:1 function accepting a float, and returning a single value.
        Multiple usages are left in the code for examples
    
    exportDescriptiveStats
        will generate exported CSVs for any number of datakeys passed to it containing descriptive stats for each of the columns specified

---------------------
Dataset avalible at https://data.world/annavmontoya/speed-dating-experiment
Dataset was collected for the paper "Gender Differences in Mate Selection: Evidence from a Speed Dating Experiment"
(Raymond Fisman, Sheena S Iyengar et al. 2006 - Quarterly Journal of Economics) avalible at: https://doi.org/10.1162/qjec.2006.121.2.673