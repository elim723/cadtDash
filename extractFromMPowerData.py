#!python3

## 
## By Elim Thompson (5/24/2022)
## 
## This script extracts information needed from PowerScribe for CADt project.
## 
## From example data, there are a total of 71 columns. I identified a few that
## seems relevant to the project. '**' means that those are especially important.
##  * Modality - str - should be mostly CT **
##  * Exam Description - str - which body part to be scanned? **
##  * Report Text - str - disease to be tested & radiologist diagnosis **
##  * Is Stat - bool - is it an stat patients? *
##  * Patient Status - str - emergency, inpatient, or outpatient? **
##  * Ordered Date - datetime - when is the scan ordered? ('2022-05-13T09:49:00.000000000')
##  * Scheduled Date - datetime - when is the scan scheduled to be?
##  * Patient Arrived Date - datetime - when the patient arrived?
##  * Exam Started Date - datetime - when the scan started?
##  * Exam Completed Date - datetime - when the scan finished? **
##  * Preliminary Report By - str - which radiologist did the preliminary report?
##  * Report Created Date - datetime - when the prelim report is started? **
##  * Preliminary Report Date - datetime - when the prelim report is completed? **
##  * Report Finalized Date - datetime - when the report is finalized by another radiologist?
##  * Exam Started to Exam Completed (minutes) - float - time difference from Exam Started Date to Exam Completed Date **
##  * Exam Completed to Report Created (minutes) - float - time difference from Exam Completed Date to Report Created Date ** 
##  * Exam Completed to Preliminary Report (minutes) - float - time difference from Exam Completed Date to Preliminary Report Date **
##  * Exam Completed to Report Finalized (minutes) - float - time difference from Exam Completed Date to Report Finalized Date **
##  * Preliminary Report to Report Finalized (minutes) - float - time difference from Preliminary Report Date to Report Finalized Date
##
## This script generates the followings:
##  - For non-datetime column, summarize with basic stats and plot histogram. 
##  - For datetime column, plot histograms of hours of the day, day of the week,
##    and month of the year
##  - For Stat, emergency, inpatient, and outpatient, plot ..?
##  - From "Exam Description", try to extract body part information
##  - From "Report Text", try to extract disease to be tested, radiologist 
##    diagnosis, and other timing information
## 
##
## 10/26/2022
## ----------
##   * Added AI call and AI use subgroup options
##   * Use the updated AiDocALLPE.xlsx for AiDoc dataframe
##   * Radiologist names and user role hisograms re-ordered
##################################################################################

###############################
## Import libraries
###############################
from http.client import REQUEST_HEADER_FIELDS_TOO_LARGE
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go

import pandas, numpy, namegenerator
from pandas.tseries.holiday import USFederalHolidayCalendar
import statsmodels.api as sm
from scipy.optimize import curve_fit, minimize
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

verbose = True

###############################
## Keyword dictionaries
###############################
BodyPartKeywords = {'chest':['chest'],
                    'abdomen':['abd', 'abdomen'],
                    'pelvis':['pelvis']}

badFitPerrThresh = 8

###############################
## Data constants
###############################
dataFilename = '../ExampleData.xlsx'
aiDocFilename = '../AiDocAllPE.xlsx'

## mPower
mPowerColumnDict = {'bool': ['Is Stat'], 'freeText': ['Report Text'],
                    'category': ['Modality', 'Patient Status',
                                 'Exam Description', 'Accession Number'],
                    'continuous': ['Exam Completed to Report Created (minutes)',
                                   'Exam Completed to Preliminary Report (minutes)',
                                   'Exam Completed to Report Finalized (minutes)',
                                   'Preliminary Report to Report Finalized (minutes)'],
                    'date': ['Exam Completed Date', 'Report Created Date',
                             'Preliminary Report Date', 'Report Finalized Date']}
mPowerColumns = mPowerColumnDict['continuous'] + mPowerColumnDict['date'] + \
                mPowerColumnDict['bool'] + mPowerColumnDict['category'] + mPowerColumnDict['freeText'] 

## AiDoc
AiDocColumns = ['Creation Date', 'Creation Time', 'Accession Number', 'Positive/Negative']

## PACS
PACSColumns = ['ACC', 'procedure', 'userfullname', 'UserRole', 'notetimestamp']

## Dropdown columns
dropdownColumns = mPowerColumnDict['date'] + \
                  mPowerColumnDict['category'] + ['AI Result', 'Decoded Radiologists', 'UserRole'] + \
                  ['Rad Open Date', 'Interopen Time (minutes)', 'Interarrival Time (minutes)', 'Mean Radiologist Service Time (minutes)']
dropdownColumns.remove ('Accession Number')
dateColumns = [col for col in dropdownColumns if 'Date' in col]
continuousColumns = [col for col in dropdownColumns if '(minutes)' in col and not col == 'Mean Radiologist Service Time (minutes)']
categoryColumns = [col for col in dropdownColumns if not '(minutes)' in col and not 'Date' in col]

## Special times
AIDoc_start_date = pandas.to_datetime ('2019-11-27 00:00')
business_hours = (8, 17) # 8am to 5pm
#  Get holidays between 2014 and 2026
holidayCal = USFederalHolidayCalendar()
holidays = holidayCal.holidays(start='2014-01-01', end='2026-12-31').to_pydatetime()
#  Threshold for interarrival time in minutes - 1 month
interarrivalTimeGapThresh = 30*24*60

binsAttempt = numpy.linspace (10, 30, 21).astype (int)
meanServiceTimeThresh = 50 # minutes

###############################
## Dataframes
###############################
get_primaryAna = lambda array: numpy.nan if len (array)==0 else array[0]
generate_random_names = lambda unique_name_list: {name:namegenerator.gen() for name in unique_name_list}

def expfunc(x, a, b, c):
    # a = amplitude, b = rate, c = cut
    return a * numpy.exp(-b*x) / (1 - numpy.exp (-b*c))

def perform_fit (values, time_thresh=numpy.inf, nbins=50, verbose=False):

    xvalues, yvalues, wvalues, has_enough = get_histo (values, nbins, time_thresh)
    
    popt = None
    good_fit = False
    bounds = [[-numpy.inf, 0], [numpy.inf, numpy.inf]]

    if has_enough:
        popt_p0_11, popt_p0_10 = None, None
        perr_p0_11, perr_p0_10 = None, None
        good_fit_p0_11, good_fit_p0_10 = False, False
        try:
            popt_p0_11, pcov = curve_fit(lambda x, a, b: expfunc(x, a, b, time_thresh), xvalues, yvalues, sigma=1/wvalues, p0=[1, 1], bounds=bounds)
            perr_p0_11 = numpy.sum (numpy.sqrt(numpy.diag(pcov)))
            if verbose:
                print ('+-----------------------------------------------------------')
                print ('| p0 = [1, 1]: {0}'.format (perr_p0_11))
            good_fit_p0_11 = True
        except:
            good_fit_p0_11 = False

        try:
            popt_p0_10, pcov = curve_fit(lambda x, a, b: expfunc(x, a, b, time_thresh), xvalues, yvalues, sigma=1/wvalues, p0=[1, 0], bounds=bounds)
            perr_p0_10 = numpy.sum (numpy.sqrt(numpy.diag(pcov)))
            if verbose: print ('| p0 = [1, 0]: {0}'.format (perr_p0_10))
            good_fit_p0_10 = True
        except:
            good_fit_p0_10 = False

        if good_fit_p0_11 and not good_fit_p0_10:
            popt, perr = popt_p0_11, perr_p0_11
        elif good_fit_p0_10 and not good_fit_p0_11:
            popt, perr = popt_p0_10, perr_p0_10
        else:
            popt = popt_p0_11 if perr_p0_11 < perr_p0_10 else popt_p0_10
            perr = min (perr_p0_11, perr_p0_10)

        good_fit = perr < badFitPerrThresh and popt[1] > 0

    if verbose: 
        text = popt[1] if good_fit else 'n/a'
        print ('| fit ({0}): {1}'.format (good_fit, text))
    if not good_fit: return None, None

    ## Calculate LLH
    ys = expfunc(xvalues, popt[0], popt[1], time_thresh)
    llh = ((ys - yvalues)**2).sum()

    return popt, llh

def get_histo (values, nbins, time_thresh):

    ## Without threshold to get normalization value
    hist, edges = numpy.histogram (values, bins=nbins)
    norm = numpy.sum (hist)

    hist, edges = numpy.histogram (values[values < time_thresh], bins=nbins)

    xvalues = edges[:-1]
    yvalues = hist 

    wvalues = numpy.sqrt (hist) 
    wvalues[numpy.where (wvalues==0)] = 1e-20

    is_valid = numpy.logical_and (numpy.logical_and (numpy.isfinite (xvalues), numpy.isfinite(yvalues)), numpy.isfinite(wvalues))

    has_enough = yvalues[is_valid] > 5
    has_enough_valid_bins = len (has_enough[has_enough]) > 3

    return xvalues[is_valid], yvalues[is_valid] / norm, wvalues[is_valid] / norm, has_enough_valid_bins

def func (r, S, N, thresh):
    return (S/N - 1/r + thresh/(numpy.exp (r*thresh)-1))**2

def get_numerical_answer (values, thresh, method='L-BFGS-B', x0=0.000001, verbose=False):

    thresh = numpy.nan_to_num (thresh)
    values = values[values < thresh]
    S = numpy.sum (values)
    N = len (values)
    res = minimize(lambda r: func(r, S, N, thresh), (x0), method=method, bounds=[[0, None]], tol=1e-6)

    if verbose: print ('| ana ({0}): {1}'.format (res.success, res.x[0]))

    if res.success and res.x[0]<1: return res.x[0]
    return None

def get_analytical_bestfit (values, time_thresh):

    anaRate = get_numerical_answer (values, time_thresh, method='L-BFGS-B', x0=0.000001)
    if anaRate is None: anaRate = get_numerical_answer (values, time_thresh, method='L-BFGS-B', x0=0.01)
    if anaRate is None: anaRate = get_numerical_answer (values, time_thresh, method='L-BFGS-B', x0=0.1)
    if anaRate is None: anaRate = get_numerical_answer (values, time_thresh, method='SLSQP', x0=0.000001)
    if anaRate is None: anaRate = get_numerical_answer (values, time_thresh, method='SLSQP', x0=0.01)
    if anaRate is None: anaRate = get_numerical_answer (values, time_thresh, method='SLSQP', x0=0.1)

    if anaRate in [1e-6, 0.01, 0.1]: return None
    return anaRate

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def find_anatomy (examDescriptions):

    ## Hold all keywords that show up in 'Exam Description'
    anatomies = {body:[] for body in BodyPartKeywords.keys()}

    ## Loop through all descriptions
    for examDes in examDescriptions:
        ## Loop through all possible anatomies
        for body, keywords in BodyPartKeywords.items():
            if pandas.isna (examDes):
                anatomies[body].append (False)
                continue
            hasBody = numpy.in1d (keywords, examDes.lower().split())
            anatomies[body].append (hasBody.any())

    ## Package anatomies dict for sub-information
    anaInfo = pandas.DataFrame (anatomies)

    ## Define a primary anatomy - ordered by anaInfo.columns
    primaryAna = [get_primaryAna (anaInfo.columns[row.values])
                  for index, row in anaInfo.iterrows()]
    
    return primaryAna, anaInfo

def is_within_business_hours (row):

    examCompletedDate = pandas.Timestamp (row['Exam Completed Date']).to_pydatetime()

    ## Not business hours if holidays
    if examCompletedDate in holidays: return False

    ## Not business hours if sunday/saturday
    if examCompletedDate.weekday() >= 5: return False

    ## Within business hours if between business_hours
    return numpy.logical_and (examCompletedDate.hour >= business_hours[0],
                              examCompletedDate.hour < business_hours[1])  

def get_truth (row):

    text = row['Report Text'].split ('\n')

    # Find the first line starts with 'PE: '
    # If addendum, the first line is the correct one.
    for line in text:
        line = line.strip()
        if len (line) == 0: continue
        ## Look for the first line that starts with 'PE:'
        if line.split ()[0].strip() == 'PE:':
            # If empty after 'PE:', assume negative
            if len (line.split()) == 1:
                return numpy.array ([None, False])
            # Record the radiologist decisions (full sentence)
            decision = ' '.join (line.split ()[1:]).strip().lower()
            if decision[-1] in ['.', ',']: decision = decision[:-1]
            # Non-Diseased if 'negative' or 'no' (not sure about indeterminate or nondiagnostic)
            values = decision.split ()
            is_not_diseased = numpy.in1d (values, ['negative', 'no', 'indeterminate', 'nondiagnostic']).any()
            return numpy.array ([decision, not is_not_diseased])

    # Some doesn't have a 'PE:' line ...
    return numpy.array (['missing', False])
    
def get_dataframe ():

    ## Read all data
    xls = pandas.ExcelFile(dataFilename)
    mpower = pandas.read_excel(xls, 'mPower', usecols=mPowerColumns, na_values='NA')
    pacs = pandas.read_excel(xls, 'CT PE Studies', usecols=PACSColumns, na_values='NA')
    xls = pandas.ExcelFile(aiDocFilename)
    aidoc = pandas.read_excel(xls, usecols=AiDocColumns, na_values='NA')

    ## Make sure all accession number columns are read as string
    mpower['Accession Number'] = mpower['Accession Number'].values.astype (str)
    pacs['ACC'] = pacs['ACC'].values.astype (str)
    aidoc['Accession Number AiDoc'] = aidoc['Accession Number'].values.astype (int).astype (str)
    aidoc = aidoc.drop (['Accession Number'], axis=1)

    ## Format datetime columns
    #  1. timestamp for radiologist open time
    pacs['Rad Open Date'] = pandas.to_datetime (pacs['notetimestamp'])
    pacs = pacs.drop (['notetimestamp'], axis=1)
    #  2. timestamps related to patients from mPower
    for timestampColumn in mPowerColumnDict['date']:
        mpower[timestampColumn] = pandas.to_datetime (mpower[timestampColumn])

    ## Merge all three frames by accession number
    ## Inner between PACS and mPower because need both info for the same patient
    merged = mpower.merge (pacs, left_on='Accession Number', right_on='ACC', how='inner')
    ## Outer between PACS/mPower and AiDoc because patients may be admitted before AiDoc is implemented
    merged = merged.merge (aidoc, left_on='Accession Number', right_on='Accession Number AiDoc', how='outer')
    ## Merged by accession number
    merged['Accession Number'] = merged['Accession Number'].combine_first (merged['Accession Number AiDoc']).combine_first (merged['ACC'])
    merged = merged.drop (['Accession Number AiDoc', 'ACC'], axis=1)

    ## Drop any rows that have AiDoc data but not PACS/mPower
    merged = merged[~merged['Modality'].isna ()]
    ## Remove cases with n/a 'Report Finalized Date' and 'Exam Completed Date' and 'Rad Open Date'
    merged = merged.dropna (axis=0, subset=['Report Finalized Date', 'Exam Completed Date', 'Rad Open Date'])

    ## For pacs, multiple entries for same acc because different radiologists read the same case
    ## Find the one that is closest to, but before, 'Report Finalized Date' in mpower
    indices = []
    accgroups = merged.groupby ('Accession Number')
    for acc in accgroups.groups:
        agroup = accgroups.get_group (acc)
        agroup = agroup[agroup['Report Finalized Date'] > agroup['Rad Open Date']]
        # Drop cases where finalized date is before all radiologist open dates
        if len (agroup) == 0: continue
        finalizedDate = agroup['Report Finalized Date'].values[0]
        indices.append ((finalizedDate - agroup['Rad Open Date']).idxmin())
    merged = merged.loc[indices]

    ## Sort entries by exam completed date i.e. timestamps when the patient enters reading list
    merged = merged.sort_values (by='Exam Completed Date')

    ## Calculate time differences
    #  1. Interarrival time
    interarrivalTime = (merged['Exam Completed Date'].values[1:] - merged['Exam Completed Date'].values[:-1])/1e9/60
    merged = merged.iloc[1:]
    merged['Interarrival Time (minutes)'] = interarrivalTime.astype (float)
    #     Exam Completed Date has a gap between 2021-12-31 and 2021-4-22.
    #     Remove the first entry after a month of gap to avoid outliers
    #     Q: What is a reasonable threshold?
    merged = merged[merged['Interarrival Time (minutes)'] < interarrivalTimeGapThresh]

    ## Add in flags / renaming columns
    #  1. Exam completed within business hours?
    merged['Within Business Hours'] = [is_within_business_hours (row) for _, row in merged.iterrows()]
    #  2. Exam being non-emergency class? 
    #merged['Is CADt Non-Emergency Class'] = merged['Patient Status'].isin (['Inpatient', 'Outpatient'])
    #  3. Patient truly diseased / non-diseased with PE?
    merged['Rad Decision'], Diseased_PE = numpy.array ([get_truth (row) for _, row in merged.iterrows()]).T
    merged['Diseased PE'] = Diseased_PE == 'True'
    #  4. Rename AI result column
    merged = merged.rename (columns={'Positive/Negative':'AI Result'})

    ## Decode readers' names
    radiologist_names = list (merged['userfullname'].value_counts().index)
    nameMap = generate_random_names (radiologist_names)
    merged['Decoded Radiologists'] = [nameMap[name] for name in merged['userfullname'].values]

    ## Get readers 
    readers = list (merged['Decoded Radiologists'].value_counts().index)

    return merged, readers

def get_interopenTimes (dataframe, time_thresh=120, meanservice_thresh=meanServiceTimeThresh):

    ## Get the list of readers in this dataframe
    readers = list (dataframe['Decoded Radiologists'].value_counts().index)

    fitRates, anaRates = [], []
    ## Perform fit per each reader
    for reader in readers:
        # Calculate inter open time
        subdf = dataframe.groupby('Decoded Radiologists').get_group (reader)
        subdf = subdf.sort_values (by='Rad Open Date')
        interopen = (subdf['Rad Open Date'].values[1:] - subdf['Rad Open Date'].values[:-1])/1e9/60
        subdf = subdf.iloc[1:]
        subdf['Interopen Time (minutes)'] = interopen.astype (float)
        
        values = subdf['Interopen Time (minutes)']
        # Method 1: perform truncated exp fit 
        nbins = find_bestfit_bins (subdf, 'Interopen Time (minutes)', time_thresh)
        fitRate, llh = perform_fit (values, nbins=nbins, time_thresh=time_thresh)
        if fitRate is not None: fitRates.append (fitRate[1])
        # Method 2: analytical solution
        anaRate = get_analytical_bestfit (values, time_thresh)
        if anaRate is not None: anaRates.append (anaRate)

    fitMeanServiceTime = 1/numpy.array (fitRates)
    fitMeanServiceTime = fitMeanServiceTime[fitMeanServiceTime<meanservice_thresh]
    anaMeanServiceTime = 1/numpy.array (anaRates)
    anaMeanServiceTime = anaMeanServiceTime[anaMeanServiceTime<meanservice_thresh]

    histdf = pandas.DataFrame(dict(
            series=numpy.concatenate((["fitting"]*len(fitMeanServiceTime), ["analytical"]*len(anaMeanServiceTime))), 
            data  =numpy.concatenate((fitMeanServiceTime,anaMeanServiceTime))
    ))
    return histdf

df, readers = get_dataframe()

###############################
## Dash app functions
###############################
def find_bestfit_bins (dataframe, selected_column, time_thresh, verbose=False):

    values = dataframe[selected_column]

    minLLH, minNbins = numpy.inf, binsAttempt[0]
    for nbins in binsAttempt:
        bestfit, llh = perform_fit (values, nbins=nbins, time_thresh=time_thresh)
        # If llh is not available, it is a bad fit i.e. can't be a minimum
        if llh is None: continue
        if verbose: print ('{0}: {1} ({2})'.format (nbins, llh, bestfit))
        if llh < minLLH:
            minLLH = llh
            minNbins = nbins
    return int (minNbins)

def generate_figure (dataframe, selected_column, nbins, yaxis_type, date_by_type, time_thresh, meanservice_thresh):

    fitTable = None
    ## holder for minifig so dash doesn't complain
    minifig = px.histogram(df, x='Interarrival Time (minutes)')
    
    if selected_column == 'Mean Radiologist Service Time (minutes)':
        series = None
        if meanservice_thresh is None: meanservice_thresh = meanServiceTimeThresh
    else:
        series = dataframe[selected_column]
        ## If no valid entries, return a histogram which will not be displayed anyways
        if len (series[~series.isna()]) == 0:
            if selected_column == 'Interopen Time (minutes)': 
                selected_column = 'Interarrival Time (minutes)'
            return px.histogram(df, x=selected_column), None, None

    if selected_column == 'Mean Radiologist Service Time (minutes)':

        histdf = get_interopenTimes (dataframe, time_thresh=time_thresh, meanservice_thresh=meanservice_thresh)
        fig = px.histogram(histdf, x='data', color='series', barmode='group', nbins=nbins)

    elif selected_column in categoryColumns:

        category_orders = None
        if selected_column == 'Decoded Radiologists':
            category_orders = dict (DecodedRadiologists=list (dataframe['Decoded Radiologists'].value_counts().index))
            category_orders['Decoded Radiologists'] = category_orders['DecodedRadiologists']
        elif selected_column == 'UserRole':
            category_orders = dict (UserRole=list (dataframe['UserRole'].value_counts().index))

        fig = px.histogram(dataframe, x=selected_column, category_orders=category_orders)

    elif selected_column in continuousColumns:

        fig = px.histogram(dataframe, x=selected_column, nbins=nbins)

        if selected_column in ['Interarrival Time (minutes)', 'Interopen Time (minutes)']:

            # Method 1: perform exp fit 
            values = dataframe[selected_column]
            bestfit, llh = perform_fit (values, nbins=nbins, time_thresh=time_thresh)
            good_fit = not bestfit is None

            hist, edges = numpy.histogram (values, bins=nbins)
            norm = numpy.sum (hist)
            hist, edges = numpy.histogram (values[values < time_thresh], bins=nbins)

            xvalues = edges[:-1]
            yvalues = hist 
            wvalues = numpy.sqrt (hist) 
            wvalues[numpy.where (wvalues==0)] = 1e-20

            is_valid = numpy.logical_and (numpy.logical_and (numpy.isfinite (xvalues), numpy.isfinite(yvalues)), numpy.isfinite(wvalues))
            xvalues, yvalues, wvalues = xvalues[is_valid], yvalues[is_valid]/norm, wvalues[is_valid]/norm

            if good_fit:
                xs = numpy.linspace(0, xvalues[-1], 1000)
                ys = expfunc(xs, bestfit[0], bestfit[1], time_thresh)

                fig = px.line (x=xs, y=ys,color_discrete_sequence=['#1f77b4'])
                error_y = go.bar.ErrorY (array=wvalues)
                fig.add_bar (x=xvalues, y=yvalues, error_y=error_y)

            else:
                error_y = go.bar.ErrorY (array=wvalues)
                fig = px.bar (x=xvalues, y=yvalues, error_y=wvalues)

            # Method 2: analytical solution
            anaRate = get_analytical_bestfit (values, time_thresh)

            # Create mini figure showing the space landscape
            S = numpy.sum (values[values < time_thresh])
            N = len (values[values < time_thresh])
            r = numpy.linspace (0, 0.1, 10000)
            if good_fit: 
                maxr = max (0.1, bestfit[1] + 0.1)
                r = numpy.linspace (0, maxr, 10000)
            thresh = numpy.nan_to_num (time_thresh)
            y = (S/N - 1/r + thresh/(numpy.exp (r*thresh)-1))**2
            is_valid = numpy.isfinite (y)
            r, y = r[is_valid], y[is_valid]
            y = y - min (y)

            minifig = go.Figure()
            minifig.add_hline(y=0, line_width=5, line_dash="solid", line_color="white")            
            minifig.add_trace(go.Scatter(x=r, 
                                             y=y, 
                                             mode='lines', 
                                             line=dict(color='#1f77b4', width=3, dash='solid'),
                                             name='likelihood space'))
            xrange = [0, max (r)]
            yrange = [-50, 500]
            
            if anaRate is not None:
                minifig.add_trace(go.Scatter(x=[anaRate, anaRate], 
                                             y=yrange, 
                                             mode='lines', 
                                             line=dict(color='green', width=3, dash='dash'),
                                             name='analytical'))
                if anaRate > xrange[1]: xrange[1] = anaRate + 0.01

            if good_fit and numpy.isfinite (bestfit[1]):
                minifig.add_trace(go.Scatter(x=[bestfit[1],bestfit[1]], 
                                             y=yrange, 
                                             mode='lines', 
                                             line=dict(color='red', width=3, dash='dash'),
                                             name='fit to truncated exp'))
                if bestfit[1] > xrange[1]: xrange[1] = bestfit[1] + 0.01
            title = 'Likelihood space' #r'$(S/N - 1/x + time_thresh/(numpy.exp (x*time_thresh)-1))^2$'
            minifig.update_layout(transition_duration=500,
                                title={'text': title,
                                        'y':0.95, 'x':0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},
                                xaxis_title='rate',
                                xaxis_range=xrange,
                                yaxis_range=yrange,
                                yaxis_title="likelihood space",
                                plot_bgcolor='rgba(0, 0, 0, 0)',
                                paper_bgcolor='rgba(0, 0, 0, 0)',
                                font_color = 'lightgray',
                                font_family = 'sans-serif',
                                font=dict(size=18))
            #minifig.update_yaxes(showticklabels=False)

            fitTable = createFitTable (bestfit, anaRate, isArrival=selected_column=='Interarrival Time (minutes)')
        
    else: ## time-series

        timeFormat = '%Y' if date_by_type == 'Year' else \
                     '%Y-%m' if date_by_type == 'Month' else \
                     '%Y-%m-%d' if date_by_type == 'Day' else \
                     '%Y-%m-%d %H' ## by hour

        subdf = series.dt.strftime (timeFormat).value_counts()
        argsort = numpy.argsort (subdf.index)
        fig = go.Figure(data=go.Scatter(x=subdf.index[argsort], 
                                        y=subdf.values[argsort],
                                        marker_color='red', text="counts"))
        fig.update_layout(yaxis_range=[0,max (subdf.values)+2])

    ylabel = "Fractional counts" if selected_column in ['Interopen Time (minutes)', 'Interarrival Time (minutes)'] else "Counts"
    fig.update_yaxes(title=ylabel,
                     type='linear' if yaxis_type == 'Linear' else 'log')

    title_suffix = ' Time Series' if selected_column in dateColumns else ' Distribution'

    fig.update_layout(transition_duration=500,
                      title={'text': selected_column + title_suffix,
                             'y':0.95, 'x':0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      xaxis_title=selected_column,
                      yaxis_title="Counts",
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      font_color = 'lightgray',
                      font_family = 'sans-serif',
                      font=dict(size=18),
                      showlegend=False)

    return fig, fitTable, minifig

def createFitTable (fit, ana, isArrival=False):

    rateText = 'Arrival rate' if isArrival else 'Service rate'
    averageTimeText = 'interarrival time' if isArrival else 'service time'

    row1 = html.Tr([html.Td(" "), html.Td('Via fitting'), html.Td('Analytical')])
    fit_value = 'n/a' if fit is None else round (fit[1], 5)
    ana_value = 'n/a' if ana is None else round (ana, 5)
    row2 = html.Tr([html.Td("{0} [1/min]: ".format (rateText)), html.Td(fit_value), html.Td(ana_value)])
    fit_value = 'n/a' if fit is None else round (1/fit[1], 2)
    ana_value = 'n/a' if ana is None else round (1/ana, 2)
    row3 = html.Tr([html.Td("Average {0} [min]:".format (averageTimeText)), html.Td(fit_value), html.Td(ana_value)])

    table_body = [html.Tbody([row1, row2, row3])]

    return dbc.Table(table_body,
                      class_name='column',
                      bordered=True,
                      color='dark',
                      hover=True,
                      responsive=True,
                      striped=True)

def generate_table(dataframe, selected_column):

    this_column = 'Decoded Radiologists' if selected_column == 'Mean Radiologist Service Time (minutes)' else selected_column
    series = dataframe[this_column]

    # 1. number of entries
    nEntries = len (series)
    row1 = html.Tr([html.Td("Number of entries"), html.Td(nEntries)])
    # 2. number of valid entries
    nValidEntries = len (series[~series.isna()])
    row2 = html.Tr([html.Td("Number of valid entries"), html.Td(nValidEntries)])

    if selected_column in categoryColumns and not selected_column == 'Mean Radiologist Service Time (minutes)':

        # 3. number of unique categories
        categoryCounts = series.value_counts()
        nCategories = len (categoryCounts.index.values)
        row3 = html.Tr([html.Td("Number of unique categories"), html.Td(nCategories)])
        # 4. category with max valid counts
        categoryMaxCount = categoryCounts.index[categoryCounts.argmax()]
        row4 = html.Tr([html.Td("Category with max valid entries"), html.Td(categoryMaxCount)])
        # 5. category with min valid counts
        categoryMinCount = categoryCounts.index[categoryCounts.argmin()]
        row5 = html.Tr([html.Td("Category with min valid entries"), html.Td(categoryMinCount)])

        table_body = [html.Tbody([row1, row2, row3, row4, row5])]

    elif selected_column in continuousColumns:

        stats = series.describe()
        # 3. mean value
        meanValue = stats.loc['mean']
        meanValue = round (meanValue, 4)
        row3 = html.Tr([html.Td("Mean value"), html.Td(meanValue)])
        # 4. range 
        arange = '({0:.2f}, {1:.2f})'.format (stats.loc['min'], stats.loc['max'])
        row4 = html.Tr([html.Td("(Min, Max) values"), html.Td(arange)])
        # 5. 25%, 50%, 75%
        percentiles = '{0:.2f}, {1:.2f}, {2:.2f}'.format (stats.loc['25%'], stats.loc['50%'], stats.loc['75%'])
        row5 = html.Tr([html.Td("25%, 50%, 75%"), html.Td(percentiles)])

        table_body = [html.Tbody([row1, row2, row3, row4, row5])]

    else:
        table_body = [html.Tbody([row1, row2])]

    return dbc.Table(table_body,
                      class_name='column',
                      bordered=True,
                      color='dark',
                      hover=True,
                      responsive=True,
                      striped=True)

def define_style (dataframe, selected_column, fitTable):

    ids = ['nbins', 'nbins-text', 'graphic-div', 'yaxis-type', 'yaxis-scale-p',
           'graphic-noValid-div', 'date-by-p', 'date-by-type', 'group-by-truth',
           'group-by-truth-p', 'group-by-AIresult', 'group-by-AIresult-p',
           'group-by-afterAI', 'group-by-afterAI-p', 'group-by-businessHours',
           'group-by-businessHours-p', 'group-by-readers', 'group-by-readers-p',
           'group-by-readers-type', 'interopen-thresh', 'interopen-thresh-p', 'nbins-option',
           'mean-service-thresh', 'mean-service-thresh-p',
           'fit-table-container', 'fit-table-container-bad-fit-p', 'minifig-div']
    styles = {anId: {'display': 'none'} for anId in ids}

    ## Make sure there are valid entries for the selected column
    series = None if selected_column == 'Mean Radiologist Service Time (minutes)' else dataframe[selected_column]
    
    if selected_column == 'Mean Radiologist Service Time (minutes)' or len (series[~series.isna()]) > 0 :
        ## Show thses elements:
        styles['graphic-div'] = {'display': 'block'} 
        styles['yaxis-type'] = {'display': 'block'}
        styles['yaxis-scale-p'] = {'display':'inline-block','margin-right':20}

        ## Show bin input only if continuous values
        if selected_column in continuousColumns and \
           not selected_column in ['Interopen Time (minutes)', 'Interarrival Time (minutes)']:
            styles['nbins'] = {'display': 'inline','width':150, 'height':35} 
            styles['nbins-text'] = {'display': 'inline-block','margin-right':20}
            styles['nbins-option'] = {'display': 'inline-block', 'margin-right':30, 'width': 200, 'margin-top':10, 'verticalAlign': 'top'}        

        ## Show date options only if datetime values
        if selected_column in dateColumns:
            styles['date-by-p'] = {'display':'inline-block','margin-right':20}
            styles['date-by-type'] = {'display': 'block'}
        
        styles['group-by-truth-p'] = {'display':'inline-block','margin-right':20}
        styles['group-by-truth'] = {'display': 'block'}

        styles['group-by-AIresult-p'] = {'display':'inline-block','margin-right':20}
        styles['group-by-AIresult'] = {'display': 'block'}

        styles['group-by-afterAI-p'] = {'display':'inline-block','margin-right':20}
        styles['group-by-afterAI'] = {'display': 'block'}

        styles['group-by-businessHours-p'] = {'display':'inline-block','margin-right':20}
        styles['group-by-businessHours'] = {'display': 'block'}           

        if selected_column == 'Decoded Radiologists':
            styles['group-by-readers-p'] = {'display':'inline-block','margin-right':20}
            styles['group-by-readers-type'] = {'display': 'block'} 

        if selected_column == 'Interopen Time (minutes)':
            styles['group-by-readers-p'] = {'display':'inline-block','margin-right':20}
            styles['group-by-readers-type'] = {'display': 'block'} 
            styles['group-by-readers'] = {'display': 'block'} 
            styles['interopen-thresh-p'] = {'display':'inline-block','margin-right':20}
            styles['interopen-thresh'] = {'display': 'block', 'height':35}

        if selected_column == 'Mean Radiologist Service Time (minutes)':
            styles['nbins'] = {'display': 'inline','width':150, 'height':35} 
            styles['nbins-text'] = {'display': 'inline-block','margin-right':20}
            styles['nbins-option'] = {'display': 'inline-block', 'margin-right':30, 'width': 200, 'margin-top':10, 'verticalAlign': 'top'}             
            styles['group-by-readers-p'] = {'display':'inline-block','margin-right':20}
            styles['group-by-readers-type'] = {'display': 'block'} 
            styles['mean-service-thresh-p'] = {'display':'inline-block','margin-right':20}
            styles['mean-service-thresh'] = {'display': 'block', 'height':35}            

        if fitTable is not None:
            styles['fit-table-container'] = {'display':'inline-block', 'width':1200}
        elif selected_column in ['Interopen Time (minutes)', 'Interarrival Time (minutes)']:
            styles['fit-table-container-bad-fit-p']  = {'display':'inline-block', "font-weight": "bold", 'fontSize':14}

        if selected_column in ['Interopen Time (minutes)', 'Interarrival Time (minutes)']:
            styles['minifig-div'] = {'display':'inline-block', 'width':1000}

    else:
        ## Show no-valid message
        styles['graphic-noValid-div'] = {'display': 'inline-block'} 

    return styles

def get_results (dataframe, readers, rad_nbins=50, rad_time_thresh=120):

    columns = ['index', 'userRole',
               'rate', 'rate_withinHour', 'rate_withinHour_diseased',
               'rate_withinHour_diseased_beforeAI', 'rate_withinHour_diseased_afterAI',
               'rate_withinHour_diseased_afterAI_AIpos', 'rate_withinHour_diseased_afterAI_AIneg',
               'rate_withinHour_nonDiseased', 'rate_withinHour_nonDiseased_beforeAI',
               'rate_withinHour_nonDiseased_afterAI', 'rate_withinHour_nonDiseased_afterAI_AIpos',
               'rate_withinHour_nonDiseased_afterAI_AIneg', 'rate_outsideHour', 'rate_outsideHour_diseased',
               'rate_outsideHour_diseased_beforeAI', 'rate_outsideHour_diseased_afterAI',
               'rate_outsideHour_diseased_afterAI_AIpos', 'rate_outsideHour_diseased_afterAI_AIneg',
               'rate_outsideHour_nonDiseased', 'rate_outsideHour_nonDiseased_beforeAI',
               'rate_outsideHour_nonDiseased_afterAI', 'rate_outsideHour_nonDiseased_afterAI_AIpos',
               'rate_outsideHour_nonDiseased_afterAI_AIneg',
               'rate_diseased', 'rate_diseased_beforeAI', 'rate_diseased_afterAI',
               'rate_diseased_afterAI_AIpos', 'rate_diseased_afterAI_AIneg',
               'rate_nonDiseased', 'rate_nonDiseased_beforeAI',
               'rate_nonDiseased_afterAI', 'rate_nonDiseased_afterAI_AIpos',
               'rate_nonDiseased_afterAI_AIneg', 
               'rate_beforeAI', 'rate_afterAI', 'rate_afterAI_AIpos', 'rate_afterAI_AIneg',
               'number', 'number_withinHour', 'number_withinHour_diseased',
               'number_withinHour_diseased_beforeAI', 'number_withinHour_diseased_afterAI',
               'number_withinHour_diseased_afterAI_AIpos', 'number_withinHour_diseased_afterAI_AIneg',
               'number_withinHour_nonDiseased', 'number_withinHour_nonDiseased_beforeAI',
               'number_withinHour_nonDiseased_afterAI', 'number_withinHour_nonDiseased_afterAI_AIpos',
               'number_withinHour_nonDiseased_afterAI_AIneg', 'number_outsideHour', 'number_outsideHour_diseased',
               'number_outsideHour_diseased_beforeAI', 'number_outsideHour_diseased_afterAI',
               'number_outsideHour_diseased_afterAI_AIpos', 'number_outsideHour_diseased_afterAI_AIneg',
               'number_outsideHour_nonDiseased', 'number_outsideHour_nonDiseased_beforeAI',
               'number_outsideHour_nonDiseased_afterAI', 'number_outsideHour_nonDiseased_afterAI_AIpos',
               'number_outsideHour_nonDiseased_afterAI_AIneg',
               'number_diseased', 'number_diseased_beforeAI', 'number_diseased_afterAI',
               'number_diseased_afterAI_AIpos', 'number_diseased_afterAI_AIneg',
               'number_nonDiseased', 'number_nonDiseased_beforeAI',
               'number_nonDiseased_afterAI', 'number_nonDiseased_afterAI_AIpos',
               'number_nonDiseased_afterAI_AIneg', 
               'number_beforeAI', 'number_afterAI', 'number_afterAI_AIpos', 'number_afterAI_AIneg']

    resultsDict = {col:[] for col in columns}

    rows = ['interarrival'] + readers

    for row in rows:

        for col in columns:

            if col == 'index':
                resultsDict['index'].append (row)
                continue

            ## User role
            if col == 'userRole':
                userRole = numpy.nan if row == 'interarrival' else dataframe[dataframe['Decoded Radiologists'] == row]['UserRole'].values[0]
                resultsDict['userRole'].append (userRole)
                continue

            ## Calculate rate  / number
            if row == 'interarrival':
                subdf = deepcopy (dataframe)
                subdf_column_name = 'Interarrival Time (minutes)'
                time_thresh = numpy.inf
                nbins = 50
            else:

                ## Get this radiologist data
                subdf = dataframe.groupby('Decoded Radiologists').get_group (row)
                subdf = subdf.sort_values (by='Rad Open Date')
                interopen = (subdf['Rad Open Date'].values[1:] - subdf['Rad Open Date'].values[:-1])/1e9/60
                subdf = subdf.iloc[1:]
                subdf['Interopen Time (minutes)'] = interopen.astype (float)
                subdf = subdf[subdf['Interopen Time (minutes)'] < rad_time_thresh]

                subdf_column_name = 'Interopen Time (minutes)'
                time_thresh = rad_time_thresh
                nbins = rad_nbins

            if '_withinHour' in col:
                subdf = subdf[subdf['Within Business Hours']]
            elif '_outsideHour' in col:
                subdf = subdf[~subdf['Within Business Hours']]   

            if '_diseased' in col:
                subdf = subdf[subdf['Diseased PE']]
            elif '_nonDiseased' in col:
                subdf = subdf[~subdf['Diseased PE']]

            if '_beforeAI' in col:
                subdf = subdf[subdf['Exam Completed Date'] < AIDoc_start_date]
            elif '_afterAI' in col:
                subdf = subdf[subdf['Exam Completed Date'] >= AIDoc_start_date] 

            if '_AIpos' in col:
                subdf = subdf[subdf['AI Result'] == 'P']
            elif '_AIneg' in col:
                subdf = subdf[subdf['AI Result'] == 'N']

            if 'rate' in col:
                rate, llh = perform_fit (subdf[subdf_column_name].values, time_thresh=time_thresh, nbins=nbins)
                resultsDict[col].append (rate)
            else: ## Number
                resultsDict[col].append (len (subdf))

    return pandas.DataFrame.from_dict (resultsDict)

def report_results (resultDataFrame):

    text = ''

    subdf = resultDataFrame[resultDataFrame['index']=='interarrival']
    subdf = subdf.loc[:, [col for col in subdf if 'rate' in col]]
    text += '+----------------------------------------------------------------------------\n'
    text += '| Average interarrival time:\n'
    for col in subdf:
        numberCol = col.replace ('rate', 'number')
        item = ' overall average time' if col == 'rate' else col.replace ('rate_', ' ').replace ('_', ' ')
        value = 'nan' if not numpy.isfinite (subdf[col].values[0]) else round (1/subdf[col].values[0], 2)
        unit = '' if not numpy.isfinite (subdf[col].values[0]) else 'min'
        text += '|   * {0:42} ({1:5}): {2} {3}\n'.format (item, resultDataFrame[numberCol][0], value, unit)
    text += '\n'

    for radType in ['Staff Radiologist', 'Resident Radiologist']:
        subdf = resultDataFrame[resultDataFrame['userRole']==radType]
        subdf = subdf.loc[:, [col for col in subdf if 'rate' in col]]

        text += '+----------------------------------------------------------------------------\n'
        text += '| Average service time by {0}:\n'.format (radType)

        for col in subdf:
            value = 'nan' if not numpy.isfinite (numpy.mean (subdf[col])) else round (1/numpy.mean (subdf[col]), 2)
            unit = '' if not numpy.isfinite (numpy.mean (subdf[col])) else 'min'
            item = ' overall average time' if col == 'rate' else col.replace ('rate_', ' ').replace ('_', ' ')
            text += '|   * {0:42}: {1} {2}\n'.format (item, value, unit)

        text += '\n'
    
    return text

###############################
## Dash app
###############################
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

app.layout = html.Div (children=[

    html.Div(
        html.H2 (children='Dashboard for FDA-Chicago CADt project')
    ),

    html.Hr(className="rounded"),

    html.Div ([

        html.Div([
            dcc.Dropdown(
                dropdownColumns,
                'Patient Status',
                id='xaxis-column'
            )
        ],style={'width': '48%', 'display': 'block'})
    ]),

    html.Hr(),

    html.Div(
        html.H3 (id='columnName-header')
    ),

    html.Div(
        id='table-container',  className='tableDiv'
    ),

    html.Br(),

    html.Div(children=[
        html.Button('Export data',id="download-csv"),
        dcc.Download(id='first_output')
    ]),

    html.Br(),

    html.Div(className='displayOptions', children=[

        html.Div (children=[
            html.Div([
                html.P('Y-scale:', id='yaxis-scale-p',
                    style={'display':'inline-block','margin-right':20}),
                dcc.RadioItems(
                    ['Linear', 'Log'],
                    'Linear',
                    id='yaxis-type',
                    inline=False,
                    labelStyle={'display': 'block'},
                    inputStyle={"margin-right": "10px", "margin-top": "5px"}
                )
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30, 'margin-top':10, 'verticalAlign': 'top'}),

        html.Div (children=[
            html.Div([
                html.P('Business Hour subgroups:', id='group-by-businessHours-p',
                    style={'display':'none'}),
                dcc.RadioItems(
                    ['All', 'Within hours', 'Outside hours'],
                    'All',
                    id='group-by-businessHours',
                    inline=False,
                    labelStyle={'display': 'block'},
                    inputStyle={"margin-right": "15px", "margin-top": "5px"},
                    style={'display':'none'}
                )
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30, 'margin-top':10, 'verticalAlign': 'top'}),

        html.Div (children=[
            html.Div([
                html.P('Truth subgroups:', id='group-by-truth-p',
                    style={'display':'none'}),
                dcc.RadioItems(
                    ['All', 'Diseased', 'Non-Diseased'],
                    'All',
                    id='group-by-truth',
                    inline=False,
                    labelStyle={'display': 'block'},
                    inputStyle={"margin-right": "15px", "margin-top": "5px"},
                    style={'display':'none'}
                )
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30, 'margin-top':10, 'verticalAlign': 'top'}),

        html.Div (children=[
            html.Div([
                html.P('AI use subgroups:', id='group-by-afterAI-p',
                    style={'display':'none'}),
                dcc.RadioItems(
                    ['All', 'Before use of AI', 'After use of AI'],
                    'All',
                    id='group-by-afterAI',
                    inline=False,
                    labelStyle={'display': 'block'},
                    inputStyle={"margin-right": "15px", "margin-top": "5px"},
                    style={'display':'none'}
                )
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30, 'margin-top':10, 'verticalAlign': 'top'}), 

        html.Div (children=[
            html.Div([
                html.P('AI call subgroups:', id='group-by-AIresult-p',
                    style={'display':'none'}),
                dcc.RadioItems(
                    ['All', 'AI Positive', 'AI Negative'],
                    'All',
                    id='group-by-AIresult',
                    inline=False,
                    labelStyle={'display': 'block'},
                    inputStyle={"margin-right": "15px", "margin-top": "5px"},
                    style={'display':'none'}
                )
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30, 'margin-top':10, 'verticalAlign': 'top'}),

        html.Div(id='nbins-option', children=[
            html.P('Enter Number of bins:', id='nbins-text',
                style={'display':'none'}),
            dcc.Input(id='nbins', placeholder='number of bins', type='number',
                    style={'display':'none'})
        ], style={'display': 'none'}),

        html.Div (children=[
            html.Div([
                html.P('Time-series by:', id='date-by-p',
                    style={'display':'none'}),
                dcc.RadioItems(
                    ['Year', 'Month', 'Day', 'Hour'],
                    'Day',
                    id='date-by-type',
                    inline=False,
                    labelStyle={'display': 'block'},
                    inputStyle={"margin-right": "15px", "margin-top": "5px"},
                    style={'display':'none'}
                )
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30, 'margin-top':10, 'verticalAlign': 'top'})

    ]),

    html.Br(),
    html.Br(),

    html.Div(className='displayOptions', children=[

        html.Div (children=[
            html.Div([
                html.P('By radiologists:', id='group-by-readers-p',
                    style={'display':'none'}),
                dcc.RadioItems(
                    ['All', 'Staff', 'Resident'],
                    'All',
                    id='group-by-readers-type',
                    inline=False,
                    labelStyle={'display': 'block'},
                    inputStyle={"margin-right": "15px", "margin-top": "5px"},
                    style={'display':'none'}
                ),
                dcc.Dropdown(
                    id='group-by-readers',
                    style={'display':'none', "margin-right": "60px", "margin-top": "5px"}
                )
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30, 'margin-top':10, 'width':250, 'verticalAlign': 'top'}),

        html.Div (children=[
            html.Div([
                html.P('Interopen Time threshold (minutes):', id='interopen-thresh-p',
                    style={'display':'none'}),
                dcc.Input(id='interopen-thresh', placeholder='threshold', type='number',
                    style={'display':'none'})
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30, 'margin-top':10, 'width':350, 'verticalAlign': 'top'}),

        html.Div (children=[
            html.Div([
                html.P('Mean Service Time threshold (minutes):', id='mean-service-thresh-p',
                    style={'display':'none'}),
                dcc.Input(id='mean-service-thresh', placeholder='threshold', type='number',
                    style={'display':'none'})
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30, 'margin-top':10, 'width':350, 'verticalAlign': 'top'})          

    ]),

    html.Br(),
    html.Br(),
    
    html.Div(
            id='fit-table-container',  className='tableDiv',
            style={'display': 'none'}
    ),

    html.Div ([
        html.P('Fit failed', id='fit-table-container-bad-fit-p',
                style={'display':'none'}),                
    ]),

    html.Br(),

    html.Div ([
        html.Div ([
            dcc.Graph(id='graphic')
        ])
    ], id='graphic-div'),

    html.Br(),

    html.Div ([
        html.Div ([
            dcc.Graph(id='minifig')
        ])
    ], id='minifig-div', style={'display': 'none'}),    

    html.Div ([
        html.P ('No valid entries for this column.')
    ], id='graphic-noValid-div', style={'display': 'none'})

])

@app.callback(
    Output('graphic', 'figure'),
    Output('minifig', 'figure'),
    Output('columnName-header', 'children'),
    Output('table-container','children'),
    Output('fit-table-container','children'),    
    Output('nbins', 'style'),
    Output('nbins-text', 'style'),
    Output('nbins-option', 'style'),
    Output('graphic-div', 'style'),  
    Output('yaxis-type', 'style'),  
    Output('yaxis-scale-p', 'style'),
    Output('date-by-type', 'style'),  
    Output('date-by-p', 'style'),      
    Output('group-by-truth', 'style'),   
    Output('group-by-truth-p', 'style'),
    Output('group-by-AIresult', 'style'),   
    Output('group-by-AIresult-p', 'style'),
    Output('group-by-afterAI', 'style'),   
    Output('group-by-afterAI-p', 'style'),    
    Output('group-by-businessHours', 'style'),
    Output('group-by-businessHours-p', 'style'),
    Output('group-by-readers', 'style'),   
    Output('group-by-readers', 'options'),
    Output('group-by-readers-type', 'style'),   
    Output('group-by-readers-p', 'style'), 
    Output('interopen-thresh', 'style'),
    Output('interopen-thresh-p', 'style'),       
    Output('mean-service-thresh', 'style'),
    Output('mean-service-thresh-p', 'style'),           
    Output('graphic-noValid-div', 'style'),    
    Output('fit-table-container','style'),
    Output('fit-table-container-bad-fit-p', 'style'),
    Output('minifig-div', 'style'),
    Input('nbins', 'value'),
    Input('xaxis-column', 'value'),
    Input('group-by-readers', 'value'),
    Input('group-by-readers-type', 'value'),
    Input('interopen-thresh', 'value'), 
    Input('mean-service-thresh', 'value'), 
    Input('yaxis-type', 'value'),
    Input('date-by-type', 'value'),
    Input('group-by-truth', 'value'),
    Input('group-by-AIresult', 'value'),
    Input('group-by-afterAI', 'value'),
    Input('group-by-businessHours', 'value')    
    )
def update_histo(nbins, selected_column, selected_reader, selected_reader_type,
                 interopen_thresh, meanservice_thresh, yaxis_type, date_by_type, group_by_truth,
                 group_by_AIresult, group_by_afterAI, group_by_busniessHours):

    dataframe = deepcopy (df)

    ## Set default values if not provided
    time_thresh = None
    readers = []
    if selected_column is None: selected_column = 'Patient Status'

    ## Get list of readers if columns can be grouped by radiologist types
    if selected_column in ['Decoded Radiologists', 'Interopen Time (minutes)', 'Mean Radiologist Service Time (minutes)']:
        if selected_reader_type == 'Staff':
            dataframe = dataframe[dataframe['UserRole'] == 'Staff Radiologist']
        elif selected_reader_type == 'Resident':
            dataframe = dataframe[dataframe['UserRole'] == 'Resident Radiologist']
        readers = list (dataframe['Decoded Radiologists'].value_counts().index)

    ## For Interopen Time, select the cases by a specific radiologist first
    if selected_column == 'Interopen Time (minutes)':
        if selected_reader is None: selected_reader = readers[0]
        dataframe = dataframe.groupby('Decoded Radiologists').get_group (selected_reader)
        dataframe = dataframe.sort_values (by='Rad Open Date')
        interopen = (dataframe['Rad Open Date'].values[1:] - dataframe['Rad Open Date'].values[:-1])/1e9/60
        dataframe = dataframe.iloc[1:]
        dataframe['Interopen Time (minutes)'] = interopen.astype (float)

    ## Handle time_thresh and nbins
    if selected_column == 'Mean Radiologist Service Time (minutes)':
        time_thresh = interopen_thresh if interopen_thresh is not None else 120

    if selected_column in ['Interopen Time (minutes)', 'Interarrival Time (minutes)']:
        time_thresh = numpy.inf if selected_column == 'Interarrival Time (minutes)' else \
                      interopen_thresh if interopen_thresh is not None else 120
        if nbins is None: nbins = find_bestfit_bins (dataframe, selected_column, time_thresh)
        dataframe = dataframe[dataframe[selected_column] < time_thresh]
        
    ## For other variables, default to 50 bins
    if nbins is None: nbins = 50

    ## If selected group by truth, slice it
    if group_by_truth == 'Diseased':
        dataframe = dataframe[dataframe['Diseased PE']]
    elif group_by_truth == 'Non-Diseased':
        dataframe = dataframe[~dataframe['Diseased PE']]

    ## If selected group by AI positive/negative, slice it
    ## Note that by selecting AI positive/negative, it is implied that data is collected after AI use.
    if group_by_AIresult == 'AI Positive':
        group_by_AIresult = 'After use of AI'
        dataframe = dataframe[dataframe['AI Result'] == 'P']
    elif group_by_AIresult == 'AI Negative':
        group_by_AIresult = 'After use of AI'
        dataframe = dataframe[dataframe['AI Result'] == 'N']

    ## If selected group by before / after use of AI, slice it
    if group_by_afterAI == 'Before use of AI':
        dataframe = dataframe[dataframe['Exam Completed Date'] < AIDoc_start_date]
    elif group_by_afterAI == 'After use of AI':
        dataframe = dataframe[dataframe['Exam Completed Date'] >= AIDoc_start_date]

    ## If selected group by busniessHours, slice it
    if group_by_busniessHours == 'Within hours':
        dataframe = dataframe[dataframe['Within Business Hours']]
    elif group_by_busniessHours == 'Outside hours':
        dataframe = dataframe[~dataframe['Within Business Hours']]

    ## Update figure
    fig, fitTable, minifig = generate_figure (dataframe, selected_column, nbins, yaxis_type, date_by_type,
                                              time_thresh=time_thresh, meanservice_thresh=meanservice_thresh)

    ## Update display
    styles = define_style (dataframe, selected_column, fitTable)

    ## Update table
    statsTable = generate_table(dataframe, selected_column)

    return fig, minifig, selected_column, statsTable, fitTable, styles['nbins'], styles['nbins-text'], styles['nbins-option'], \
           styles['graphic-div'], styles['yaxis-type'], styles['yaxis-scale-p'], \
           styles['date-by-type'], styles['date-by-p'], styles['group-by-truth'], \
           styles['group-by-truth-p'], styles['group-by-AIresult'], \
           styles['group-by-AIresult-p'], styles['group-by-afterAI'], \
           styles['group-by-afterAI-p'], styles['group-by-businessHours'], \
           styles['group-by-businessHours-p'], styles['group-by-readers'], readers, styles['group-by-readers-type'], \
           styles['group-by-readers-p'], styles['interopen-thresh'], \
           styles['interopen-thresh-p'], styles['mean-service-thresh'], \
           styles['mean-service-thresh-p'], styles['graphic-noValid-div'], \
           styles['fit-table-container'], styles['fit-table-container-bad-fit-p'], styles['minifig-div']

@app.callback(
    Output("first_output","data"),
    Input('download-csv','n_clicks'), 
    prevent_initial_call = True,
)
def dl_un_csv(n_clicks):
    results = get_results (df, readers, rad_nbins=20, rad_time_thresh=120)
    return dcc.send_data_frame(results.to_csv, filename=f'results.csv',sep=',',header=True,index=False,encoding='utf-8')


if __name__ == '__main__':

    app.run_server(debug=True)

