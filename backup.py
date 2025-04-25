import dash
from plotly import graph_objs as go
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
from scipy.stats import norm
import requests
import dash_bootstrap_components as dbc

#Load Data

#Loading NSE Option Chain Data by Scrapping

# symbol = ['NIFTY', 'BANKNIFTY,]

def fetch_nse_option_chain(symbol='NIFTY'):
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.nseindia.com/option-chain"
    }


    session = requests.Session()
    session.headers.update(headers)

    # Initial request to get cookies
    session.get("https://www.nseindia.com")
    # session = session.get(url)

    #Fetch data
    response = session.get(url)
    data = response.json()

    spot_price = data['records']['underlyingValue']
    records = []

    for item in data['records']['data']:
        strike = item.get('strikePrice')
        expiry = item.get('expiryDate')

        ce = item.get('CE', {})
        pe = item.get('PE', {})

        if ce:
            records.append({
                'Strike': strike,
                'Type': 'CALL',
                'LTP': ce.get('lastPrice'),
                'IV': ce.get('impliedVolatility'),
                'Volume': ce.get('totalTradedVolume'),
                'OI': ce.get('openInterest'),
                'Expiry': expiry
            })

        if pe:
            records.append({
                'Strike': strike,
                'Type': 'PUT',
                'LTP': pe.get('lastPrice'),
                'IV': pe.get('impliedVolatility'),
                'Volume': pe.get('totalTradedVolume'),
                'OI': pe.get('openInterest'),
                'Expiry': expiry
            })

    df = pd.DataFrame(records)
    df['Underlying'] = spot_price
    df['DaysToExpiry'] = (pd.to_datetime(df['Expiry'], dayfirst=False) - pd.Timestamp.today()).dt.days

    return df, spot_price


data,spot_price = fetch_nse_option_chain()
#to find the atm stirke in data table
atm_strike = data['Strike'].iloc[(data['Strike'] - spot_price).abs().argsort()[0]]
# to set the value for min drop down
atm_strike_2 = data.loc[(data['Strike'] - spot_price).abs().idxmin(), 'Strike']


#Checking the data
print(data.head())


#Black Scholes model
def calculate_greeks(row, r=0.05):
    S = row['Underlying']      # Spot price
    K = row['Strike']          # Strike price
    T = row['DaysToExpiry'] / 365  # Time to expiry in years
    sigma = row['IV'] / 100    # Convert IV to decimal
    type_ = row['Type']

    if T <= 0 or sigma <= 0:
        return pd.Series([np.nan] * 4, index=['Delta', 'Gamma', 'Vega', 'Theta'])

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type_ == 'CALL':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return pd.Series([delta, gamma, vega / 100, theta / 365], index=['Delta', 'Gamma', 'Vega', 'Theta'])  # daily theta

if all(col not in data.columns for col in ['Delta', 'Gamma', 'Theta', 'Vega']):
    greeks = data.apply(calculate_greeks, axis=1)
    data = pd.concat([data, greeks], axis=1)

pd.set_option('display.max_columns', None)
# print(data)

#Calculate max pain function

def calculate_max_pain(df, spot_price):
    unique_strike = df['Strike'].unique()
    pain_points = []

    for strike in unique_strike:
        total_pain = 0

        for i in range (len(df)):
            row = df.iloc[i]
            row_strike = row["Strike"]
            option_type = row['Type']
            oi = row['OI']

            if option_type == 'CALL':
                call_pain = max(row_strike - strike, 0) * oi
                total_pain += call_pain

            elif option_type == "PUT":
                put_pain = max(strike - row_strike, 0) * oi
                total_pain += put_pain

        pain_points.append((strike, total_pain))

    pain_df = pd.DataFrame(pain_points, columns=['Strike', 'TotalPain'])
    # max_pain_strike = min(pain_points, key=lambda x: x[1])[0]
    max_pain_strike = pain_df.loc[pain_df['TotalPain'].idxmin(), 'Strike']
    return max_pain_strike, pain_df

max_pain, pain_df = calculate_max_pain(data,spot_price)

# print(pain_df)

#Creating PCR Data Frame
pcr_data = data[['Strike', 'Type', 'OI', 'Expiry']].copy()
pcr_grouped = pcr_data.groupby(['Strike', 'Expiry', 'Type'])['OI'].sum().unstack().reset_index()
pcr_grouped.columns.Name = None # removing Type index label
pcr_grouped = pcr_grouped.rename(columns={'PUT': 'OI_PUT', 'CALL': 'OI_CALL'})
pcr_grouped.dropna(subset=['OI_PUT','OI_CALL'], inplace = True)
pcr_grouped['PCR'] = pcr_grouped['OI_PUT'] / pcr_grouped['OI_CALL']
pcr_grouped.dropna(subset=['PCR'], inplace = True)
# print(pcr_grouped)

print(f'Max Pain is: {max_pain}')



#Initialize the dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Options Analytics Dashboard"

option_chain_columns = ['Strike', 'Type', 'LTP', 'IV', 'Volume', 'OI', 'Expiry']

#Layout

app.layout =  dbc.Container([
        html.H1("📊 F&O Options Analytics Dashboard",
                style={'textAlign': 'center', 'padding': '10px'}),

        #Select Options
        dbc.Row([
            # dbc.Col([
            #     html.Label('Select Index Options:'),
            #     dcc.Dropdown(
            #         id='index-symbols',
            #         options=[
            #             {'label':'NIFTY','value':'NIFTY'},
            #             {'label':'BANKNIFTY','value':'BANKNIFTY'},
            #             {'label':'FINNIFTY','value':'FINNIFTY'},
            #         ],
            #         value='NIFTY',
            #         clearable=False,
            #         style={'width':'150px'})
            # ], width=2),

            # Call-Put-ALl options
            dbc.Col([
                html.Label("Select Option Type:"),
                dcc.Dropdown(
                id='option-type',
                options = [
                    {'label':'CALL', 'value': 'CALL'},
                    {'label':'PUT', 'value': 'PUT'},
                    {'label':'ALL', 'value': 'ALL'}
                ],
                value = 'ALL',
                clearable = False,
                style = {'width':'150px'}),
            ], width=2),

            dbc.Col([
            html.Label("Select Expiry:"),
            dcc.Dropdown(
                id="expiry-filter",
                options=[{'label': i, 'value': i} for i in sorted(data['Expiry'].unique())],
                # value= data['Expiry'].unique()[0],
                value= data['Expiry'].min(),
                clearable=False,
                style= {'width': '150px'})
            ], width=2),

            # Min & Max Divs
            dbc.Col([
            html.Label("Select Min Strike"),
            dcc.Dropdown(
                id="min-strike",
                options=[{'label': i, 'value': i} for i in sorted(data['Strike'].drop_duplicates())],
                value = atm_strike_2 - 200,
                clearable= True,
                style= {'width': '150px'}),
            ],width=2),

            dbc.Col([
            html.Label("Select Max Strike"),
            dcc.Dropdown(
                id="max-strike",
                options=[{'label': i, 'value': i} for i in sorted(data['Strike'].drop_duplicates())],
                value = data['Strike'].max(),
                clearable= True,
                style= {'width': '150px'}),
            ],width=2),

            # Stats Rows (Section)
            dbc.Col([
                html.Div(id='metrics-display',className="border p-2 bg-light",
                         style= {'fontWeight':'bold'})
            ], width=4),
        ], className="mb-3"), # dbc row

        #Tabs Section for Table and Graphs and Greeks
        dbc.Tabs([
            dbc.Tab(label= 'Option Chain Table', children=[
            dash_table.DataTable(
                id='option-table',
                columns=[{"name": i, "id": i} for i in option_chain_columns],
                data = data.to_dict("records"),
                page_size=20,
                style_table={'overflowX':'auto'},
                style_cell={'textAlign':'center'},
                style_header={'backgroundColor': '#f1f1f1', 'fontWeight':'bold'},
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{{Strike}} = {}'.format(atm_strike)
                        },
                        'backgroundColor': '#d1f0ff',
                        'color': 'black',
                        'fontWeight': 'bold'
                    }]
                )
            ]),

            #Graphs
            dcc.Tab(label='Charts', children=[
                dcc.Graph(id='oi-strike-chart'),
                # dcc.Graph(id='io_change_chart'),
                dcc.Graph(id='iv-volume-chart'),
                dcc.Graph(id='max_pain_graph'),
                dcc.Graph(id='PCR-Chart')]),

            #Greeks Table
            dcc.Tab(label='Option Greeks',children=[
                dash_table.DataTable(
                    id='greeks-table',
                    columns = [{'name': i, 'id': i} for i in ['Strike', 'Type', 'Delta','Gamma', 'Vega', 'Theta']],
                    data = [],
                    page_size = 10,
                    style_table = {'overflowX': 'auto'},
                    style_cell = {'textAlign': 'center'},
                    style_header = {'backgroundColor': '#f1f1f1', 'fontWeight':'bold'},)]),
        ], style={'padding': '10px'}),

        #Download button
        dbc.Row([
            dbc.Col(
                dbc.Button("Download CSV", id="download_button", color="primary", className="mb-2"),
                width="auto")
            ]), dcc.Download(id="download-dataframe_csv")

    ],fluid=True)

#Callback to filter table based on dropdown
@app.callback(
    [dash.dependencies.Output('option-table', 'data'),
    dash.dependencies.Output('metrics-display', 'children'),
    dash.dependencies.Output('oi-strike-chart', 'figure'),
    # dash.dependencies.Output('io_change_chart', 'figure'),
    dash.dependencies.Output('iv-volume-chart', 'figure'),
    dash.dependencies.Output('max_pain_graph', 'figure'),
    dash.dependencies.Output('PCR-Chart', 'figure'),
    dash.dependencies.Output('greeks-table', 'data')],
    [dash.dependencies.Input('option-type', 'value'),
     dash.dependencies.Input('expiry-filter', 'value'),
     dash.dependencies.Input('min-strike', 'value'),
     dash.dependencies.Input('max-strike', 'value'),]
)

def update_table(selected_type,selected_expiry,min_strike, max_strike):

    filtered_data = data[data['Expiry'] == selected_expiry].copy()
    if selected_type != 'ALL':
        filtered_data = filtered_data[filtered_data['Type'] == selected_type]
    filtered_data = filtered_data[
        (filtered_data['Strike'] >= min_strike) &
        (filtered_data['Strike'] <= max_strike)
        ]
    total_oi = round(filtered_data['OI'].sum(),4)
    avg_iv = round(filtered_data['IV'].mean(),4)
    most_active_strike = filtered_data.loc[filtered_data['OI'].idxmax(),'Strike']
    most_active_strike_type = filtered_data.loc[filtered_data['OI'].idxmax(),'Type']
    metrics = [
        html.Div(f"Total OI: {total_oi}"),
        html.Div(f"Average IV: {avg_iv:.2f}"),
        html.Div(f"Most Active Strike: {most_active_strike} {most_active_strike_type}"),
        html.Div(f"Current Spot: {spot_price}"),
        html.Div(f"Max Pain Point: {max_pain}"),
    ]

    line_fig = {
        'data': [
            {'x': filtered_data['Strike'],'y': filtered_data['OI'], 'type': 'line', 'name': 'OI'}
        ],
        'layout': {
            'title': {'text': 'Open Interest vs Strike', 'x': 0.5},
            'xaxis': {'title': 'Strike', 'tickformat': ',d'},
            'yaxis': {'title': 'Open Interest'}
        }
    }
    #Bar Chart
    bar_fig = {
    'data': [
        {'x': filtered_data['Strike'],'y': filtered_data['IV'], 'type': 'bar', 'name': 'IV'},
        {'x': filtered_data['Strike'], 'y': filtered_data['Volume'], 'type': 'bar', 'name': 'Volume'}
    ],
        'layout' : {
            'title': {'text': 'IV & Volume by Strike', 'x': 0.5},
            'xaxis': {'title': 'Strike', 'tickformat': ',d'},
            'yaxis': {'title': 'Value'},
            'barmode': 'group'
        }
    }

    #Max Pain Graph
    fig_max_pain = go.Figure()

    fig_max_pain.add_trace(go.Scatter(
        x = pain_df['Strike'], y= pain_df['TotalPain'],
        mode='lines+markers',
        name='Total Pain',
        line=dict(color='royalblue')))

    fig_max_pain.add_trace(go.Scatter(
        x = [max_pain], y = [pain_df[pain_df['Strike'] == max_pain]['TotalPain'].values[0]],
        mode = 'markers+text',
        name= 'Max Pain Point',
        marker= dict(color='red', size=10),
        text=[f'Max Pain: {max_pain}'],
        textposition= 'top center'
    ))

    fig_max_pain.update_layout(
        title='Max Pain vs Strike Price',
        xaxis = dict(
            title = 'Strike Price',
            tickformat = ',d'
        ),
        yaxis_title='Total Pain',
        template='plotly_white',
        height=400
    )

    figure_pcr = {
        'data': [
            go.Scatter(
                x = pcr_grouped['Strike'],
                y = pcr_grouped['PCR'],
                mode= 'lines+markers',
                name='PCR',
                line=dict(color='blue', width=2)),

            go.Scatter(
                x = pcr_grouped['Strike'],
                y=[1] * len(pcr_grouped),  # Horizontal line at PCR=1
                mode ='lines',
                name='Neutral PCR',
                line=dict(color='gray', dash='dash'))
            ],
            'layout': go.Layout(
                title = 'Put Call Ratio',
                xaxis={'title': 'Strike Price', 'tickformat': ',d'},
                yaxis={'title':'PCR'},
                hovermode='closest',
                height=400
            )
    }

    filtered_data['Delta'] = filtered_data['Delta'].round(4)
    filtered_data['Gamma'] = filtered_data['Gamma'].round(4)
    filtered_data['Vega'] = filtered_data['Vega'].round(4)
    filtered_data['Theta'] = filtered_data['Theta'].round(4)

    greek_cols = ['Strike','Type', 'Delta', 'Gamma', 'Vega', 'Theta']
    greek_data = filtered_data[greek_cols].to_dict('records')


    return filtered_data.to_dict('records'), metrics, line_fig, bar_fig, fig_max_pain, figure_pcr, greek_data

# def update_oi_charts(selected_option_type, selected_expiry):
#     filtered_data_oi = data[
#         (data['Type'] == selected_option_type) &
#         (data['Expiry'] == selected_expiry)
#     ]
#
#     #generating random oi change values
#     np.random.seed(42)
#     filtered_data_oi['OI_Change'] = np.random.randint(-1000, 1000, size=len(filtered_data_oi))
#
#
#     #OI Chart
#
#     oi_fig = go.Figure()
#
#     oi_fig.addTrace(go.Bar(
#         x = filtered_data_oi['Strike'],
#         y = filtered_data_oi['OI'],
#         name='OI',
#         marker_color='indigo')),
#
#     oi_fig.update_layout(
#         title='Open Interest vs Strike Price',
#         xaxis_title='Strike Price',
#         yaxis_title='Open Interest',
#         template='plotly_white'),
#
#     #Change in OI charts
#     change_fig = go.Figure()
#     change_fig.add_trace(go.Bar(
#         x = filtered_data_oi['Strike'],
#         y = filtered_data_oi['OI_Change'],
#         name = 'Change in OI',
#         marker_color = 'darkorange'))
#
#     change_fig.update_layout(
#         title='Change in Open Interest vs Strike Price',
#         xaxis_title= {'title': 'Strike Price', 'tickformat': ',d'},
#         yaxis_title='Change in Open Interest',
#         template='plotly_white')
#
#
#     return oi_fig, change_fig



@app.callback(
    dash.dependencies.Output('download-dataframe_csv', 'data'),
    dash.dependencies.Input('download_button', 'n_clicks'),
    [dash.dependencies.State('option-type', 'value'),
     dash.dependencies.State('expiry-filter', 'value'),
     dash.dependencies.State('min-strike', 'value'),
     dash.dependencies.State('max-strike', 'value'),],
    prevent_initial_call = True
)

def download_filtered_data(n_clicks, selected_type, selected_expiry, min_strike, max_strike):
    filtered_data = data[data['Expiry'] == selected_expiry].copy()
    if selected_type != 'ALL':
        filtered_data = filtered_data[filtered_data['Type'] == selected_type]
    filtered_data = filtered_data[
        (filtered_data['Strike'] >= min_strike) &
        (filtered_data['Strike'] <= max_strike)
    ]
    return dcc.send_data_frame(filtered_data.to_csv, "filtered_option_data.csv", index=False)


#Run Server
if __name__ == '__main__':
    # print(data.head())
    app.run(debug=True, use_reloader=False)

