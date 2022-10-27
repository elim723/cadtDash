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
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

import pandas, numpy
from pandas.tseries.holiday import USFederalHolidayCalendar
from copy import deepcopy

###############################
## Keyword dictionaries
###############################
BodyPartKeywords = {'chest':['chest'],
                    'abdomen':['abd', 'abdomen'],
                    'pelvis':['pelvis']}

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
dropdownColumns = mPowerColumnDict['continuous'] + mPowerColumnDict['date'] + \
                  mPowerColumnDict['category'] + ['AI Result', 'userfullname', 'UserRole'] + \
                  ['Rad Open Date', 'Rad Read Duration (minutes)', 'Interarrival Time (minutes)']
dateColumns = [col for col in dropdownColumns if 'Date' in col]
continuousColumns = [col for col in dropdownColumns if '(minutes)' in col]
categoryColumns = [col for col in dropdownColumns if not '(minutes)' in col and not 'Date' in col]

## Special times
AIDoc_start_date = pandas.to_datetime ('2019-11-27 00:00')
business_hours = (8, 17) # 8am to 5pm
#  Get holidays between 2014 and 2026
holidayCal = USFederalHolidayCalendar()
holidays = holidayCal.holidays(start='2014-01-01', end='2026-12-31').to_pydatetime()
#  Threshold for interarrival time in minutes - 1 month
interarrivalTimeGapThresh = 30*24*60

###############################
## Dataframes
###############################
get_primaryAna = lambda array: numpy.nan if len (array)==0 else array[0]

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
    #  1. Radiologist reading time
    merged['Rad Read Duration (minutes)'] = (merged['Report Finalized Date'] - merged['Rad Open Date']).dt.total_seconds()/60
    #  2. Interarrival time
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
    merged['Is CADt Non-Emergency Class'] = merged['Patient Status'].isin (['Inpatient', 'Outpatient'])
    #  3. Patient truly diseased / non-diseased with PE?
    merged['Rad Decision'], Diseased_PE = numpy.array ([get_truth (row) for _, row in merged.iterrows()]).T
    merged['Diseased PE'] = Diseased_PE == 'True'
    #  4. Rename AI result column
    merged = merged.rename (columns={'Positive/Negative':'AI Result'})

    return merged

df = get_dataframe()

###############################
## Dash app functions
###############################
def generate_figure (dataframe, selected_column, nbins, yaxis_type, date_by_type):

    ## If no valid entries, return a histogram which will not be displayed anyways
    series = dataframe[selected_column]
    if len (series[~series.isna()]) == 0:
        return px.histogram(df, x=selected_column) 

    if selected_column in categoryColumns:

        category_orders = None
        if selected_column == 'userfullname':
            category_orders = dict (userfullname=list (dataframe['userfullname'].value_counts().index))
        elif selected_column == 'UserRole':
            category_orders = dict (UserRole=list (dataframe['UserRole'].value_counts().index))

        fig = px.histogram(dataframe, x=selected_column, category_orders=category_orders)

    elif selected_column in continuousColumns:
        fig = px.histogram(dataframe, x=selected_column, nbins=nbins)

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

    fig.update_yaxes(title="Counts",
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
                      font_family = 'sans-serif')

    return fig    

def generate_table(dataframe, selected_column):

    series = dataframe[selected_column]

    # 1. number of entries
    nEntries = len (series)
    row1 = html.Tr([html.Td("Number of entries"), html.Td(nEntries)])
    # 2. number of valid entries
    nValidEntries = len (series[~series.isna()])
    row2 = html.Tr([html.Td("Number of valid entries"), html.Td(nValidEntries)])

    if selected_column in categoryColumns:

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
        meanValue = round (stats.loc['mean'], 4)
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

def define_style (dataframe, selected_column):

    ids = ['nbins', 'nbins-text', 'graphic-div', 'yaxis-type', 'yaxis-scale-p',
           'graphic-noValid-div', 'date-by-p', 'date-by-type', 'group-by-truth',
           'group-by-truth-p', 'group-by-AIresult', 'group-by-AIresult-p',
           'group-by-afterAI', 'group-by-afterAI-p', 'group-by-businessHours',
           'group-by-businessHours-p', 'group-by-cadtClass', 'group-by-cadtClass-p']
    styles = {anId: {'display': 'none'} for anId in ids}

    ## Make sure there are valid entries for the selected column
    series = dataframe[selected_column]
    
    if len (series[~series.isna()]) > 0 :
        ## Show thses elements:
        styles['graphic-div'] = {'display': 'block'} 
        styles['yaxis-type'] = {'display': 'block'}
        styles['yaxis-scale-p'] = {'display':'inline-block','margin-right':20}

        ## Show bin input only if continuous values
        if selected_column in continuousColumns:
            styles['nbins'] = {'display': 'inline','width':150, 'height':20} 
            styles['nbins-text'] = {'display': 'inline-block','margin-right':20}

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

        styles['group-by-cadtClass-p'] = {'display':'inline-block','margin-right':20}
        styles['group-by-cadtClass'] = {'display': 'block'} 

    else:
        ## Show no-valid message
        styles['graphic-noValid-div'] = {'display': 'inline-block'} 

    return styles

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
        ], style={'display': 'inline-block', 'margin-right':30}),

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
        ], style={'display': 'inline-block', 'margin-right':30}),

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
        ], style={'display': 'inline-block', 'margin-right':30}),

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
        ], style={'display': 'inline-block', 'margin-right':30}),

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
        ], style={'display': 'inline-block', 'margin-right':30}),

        html.Div (children=[
            html.Div([
                html.P('CADt nonEmergency subgroups:', id='group-by-cadtClass-p',
                    style={'display':'none'}),
                dcc.RadioItems(
                    ['All', 'Emergency', 'Non-Emergency'],
                    'All',
                    id='group-by-cadtClass',
                    inline=False,
                    labelStyle={'display': 'block'},
                    inputStyle={"margin-right": "15px", "margin-top": "5px"},
                    style={'display':'none'}
                )
            ],style={'display': 'block'})
        ], style={'display': 'inline-block', 'margin-right':30}),        

        html.Div(children=[
            html.P('Enter Number of bins:', id='nbins-text',
                style={'display':'none'}),
            dcc.Input(id='nbins', placeholder='number of bins', type='number',
                    style={'display':'none'})
        ], style={'display': 'inline-block', 'width': 200}),

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
        ], style={'display': 'inline-block', 'margin-right':30})

    ]),

    html.Br(),

    html.Div ([
        html.Div ([
            dcc.Graph(id='graphic')
        ])
    ], id='graphic-div'),

    html.Div ([
        html.P ('No valid entries for this column.')
    ], id='graphic-noValid-div', style={'display': 'none'})

])

@app.callback(
    Output('graphic', 'figure'),
    Output('columnName-header', 'children'),
    Output('table-container','children'),
    Output('nbins', 'style'),
    Output('nbins-text', 'style'),
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
    Output('group-by-cadtClass', 'style'),   
    Output('group-by-cadtClass-p', 'style'),  
    Output('graphic-noValid-div', 'style'),  
    Input('nbins', 'value'),
    Input('xaxis-column', 'value'),
    Input('yaxis-type', 'value'),
    Input('date-by-type', 'value'),
    Input('group-by-truth', 'value'),
    Input('group-by-AIresult', 'value'),
    Input('group-by-afterAI', 'value'),
    Input('group-by-businessHours', 'value'),
    Input('group-by-cadtClass', 'value')
    )
def update_histo(nbins, selected_column, yaxis_type, date_by_type, group_by_truth,
                 group_by_AIresult, group_by_afterAI, group_by_busniessHours, group_by_CADtClass):

    ## Set default values if not provided
    if selected_column is None: selected_column = 'Patient Status'
    if nbins is None: nbins = 10

    ## If selected group by truth, slice it
    dataframe = deepcopy (df)
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

    ## If selected group by cadtClass, slice it
    if group_by_CADtClass == 'Non-Emergency':
        dataframe = dataframe[dataframe['Is CADt Non-Emergency Class']]
    elif group_by_CADtClass == 'Emergency':
        dataframe = dataframe[~dataframe['Is CADt Non-Emergency Class']]

    ## Update figure
    fig = generate_figure (dataframe, selected_column, nbins, yaxis_type, date_by_type)

    ## Update display
    styles = define_style (dataframe, selected_column)

    ## Update table
    statsTable = generate_table(dataframe, selected_column)

    return fig, selected_column, statsTable, styles['nbins'], styles['nbins-text'], \
           styles['graphic-div'], styles['yaxis-type'], styles['yaxis-scale-p'], \
           styles['date-by-type'], styles['date-by-p'], styles['group-by-truth'], \
           styles['group-by-truth-p'], styles['group-by-AIresult'], \
           styles['group-by-AIresult-p'], styles['group-by-afterAI'], \
           styles['group-by-afterAI-p'], styles['group-by-businessHours'], \
           styles['group-by-businessHours-p'], styles['group-by-cadtClass'], \
           styles['group-by-cadtClass-p'], styles['graphic-noValid-div']

if __name__ == '__main__':
    app.run_server(debug=True)

