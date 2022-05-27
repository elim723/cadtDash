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
##################################################################################

###############################
## Import libraries
###############################
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import pandas, numpy

###############################
## Dataframe
###############################
mPowerDataFilename = '../mPowerExampleData.xlsx'
boolColumns = ['Is Stat']
categoryColumns = ['Modality', 'Patient Status', 'Preliminary Report By', 'Exam Description']
freeTextColumns = ['Report Text']
continuousColumns = ['Exam Started to Exam Completed (minutes)',
                     'Exam Completed to Report Created (minutes)',
                     'Exam Completed to Preliminary Report (minutes)',
                     'Exam Completed to Report Finalized (minutes)',
                     'Preliminary Report to Report Finalized (minutes)']
dateColumns = ['Ordered Date', 'Scheduled Date', 'Patient Arrived Date',
               'Exam Started Date', 'Exam Completed Date', 'Report Created Date',
               'Preliminary Report Date', 'Report Finalized Date']
columns = continuousColumns + dateColumns + boolColumns + categoryColumns

df = pandas.read_excel(mPowerDataFilename, usecols=columns, na_values='NA')

def massage_dataframe (df):

    return

###############################
## Dash app functions
###############################
def generate_figure (dataframe, selected_column, nbins, yaxis_type, date_by_type):

    ## If no valid entries, return a histogram which will not be displayed anyways
    series = df[selected_column]
    if len (series[~series.isna()]) == 0:
        return px.histogram(df, x=selected_column) 

    if selected_column in categoryColumns:
        fig = px.histogram(dataframe, x=selected_column)

    elif selected_column in continuousColumns:
        fig = px.histogram(dataframe, x=selected_column, nbins=nbins)

    else: ## time-series

        timeFormat = '%Y' if date_by_type == 'Year' else \
                     '%Y-%m' if date_by_type == 'Month' else \
                     '%Y-%m-%d' if date_by_type == 'Day' else \
                     '%Y-%m-%d %H' ## by hour

        subdf = df[selected_column].dt.strftime (timeFormat).value_counts()
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
           'graphic-noValid-div', 'date-by-p', 'date-by-type']
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
                continuousColumns + dateColumns + categoryColumns,
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
        ], style={'display': 'inline-block', 'width':170}),

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
    Output('graphic-noValid-div', 'style'),  
    Input('nbins', 'value'),
    Input('xaxis-column', 'value'),
    Input('yaxis-type', 'value'),
    Input('date-by-type', 'value')
    )
def update_histo(nbins, selected_column, yaxis_type, date_by_type):

    ## Set default values if not provided
    if selected_column is None: selected_column = 'Patient Status'
    if nbins is None: nbins = 10

    ## Update figure
    fig = generate_figure (df, selected_column, nbins, yaxis_type, date_by_type)

    ## Update display
    styles = define_style (df, selected_column)

    ## Update table
    statsTable = generate_table(df, selected_column)

    return fig, selected_column, statsTable, styles['nbins'], styles['nbins-text'], \
           styles['graphic-div'], styles['yaxis-type'], styles['yaxis-scale-p'], \
           styles['date-by-type'], styles['date-by-p'], styles['graphic-noValid-div']

if __name__ == '__main__':
    app.run_server(debug=True)

