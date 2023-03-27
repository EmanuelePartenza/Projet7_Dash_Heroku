# https://dash.plotly.com/dash-core-components/dropdown#multi-value-dropdown
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash import Input, Output
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import requests
import plotly.express as px


# LOAD DATA
#
# proba
predictions = np.load("./all_proba.npy")
all_proba = pd.DataFrame()
all_proba['prob'] = predictions
# feature importance
feat_importance = pd.read_csv('feature_importance_df.csv')  # , index_col='feature')
feat_importance = feat_importance.sort_values('importance')  # , ascending=False)

# DataFrame
df = pd.read_csv("./DataFrame.csv", index_col='SK_ID_CURR')
# Sample
X_test_sample = pd.read_csv("./X_test_sample.csv", index_col='SK_ID_CURR')
id_list = X_test_sample.index.tolist()


# "C:\Users\emanu\PycharmProjects\pythonProject\Projet7_Dashapp\DataFrame.zip"


def predict_dash(pred: dict):
    if pred['classe_solvab'] == 0:
        value = pred['proba_classe_0']
        fig = go.Figure(go.Indicator(
            mode="delta+gauge+number",
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': 'green'},
                   },
            delta={"reference": 50},
            value=value,
            title={'text': "prediction"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
    elif pred['classe_solvab'] == 1:
        value = pred['proba_classe_0']
        fig = go.Figure(go.Indicator(
            mode="delta+gauge+number",
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': 'red'},
                   },
            delta={"reference": 50},
            value=value,
            title={'text': "prediction"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
    fig.update_layout(autosize=False,
                      width=500,
                      height=500,
                      margin=dict(l=40, r=40, t=40, b=40))
    return fig


def feature_dash(feat_imp):
    feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1]))
    trace1 = {
        "type": "bar",
        "x": list(feat_imp.values())[-20:],
        "y": list(feat_imp.keys())[-20:],
        "orientation": "h"
    }
    data = go.Data([trace1])
    layout = {
        "title": {"text": "Feature Importance"},
        "width": 800,
        "xaxis": {"title": {"text": "Feature Importance"}},
        "yaxis": {
            "title": {
                "font": {"size": 16},
                "text": "features"
            },
            "tickfont": {"size": 10}
        },
        "height": 800,
        "margin": {"l": 200},
        "autosize": False
    }
    fig = go.Figure(data=data, layout=layout)
    return fig


def show_more_rilevant_feat(n, value):
    costumer_id = value

    feature_list = feat_importance.tail(n)['feature'].to_list()

    fig = make_subplots(rows=n, cols=1, subplot_titles=feature_list)

    color_list = ['violet', 'red', 'blue', 'green', 'yellow', 'salmon', 'purple', 'gold', 'orange', 'olive', 'cyan']
    i = 0
    for feature in feature_list:
        i += 1
        trace = ff.create_distplot([df[feature].to_list()], [feature],
                                   [df[feature].max() / 20])  # histnorm='probability')

        fig.add_trace(go.Histogram(trace['data'][0],
                                   marker_color=color_list[i-1],
                                   text=feature
                                   ),
                      row=i, col=1
                      )
        fig.add_shape(go.layout.Shape(type='line', xref='x', yref='y2 domain',
                                      y0=0,
                                      x0=float(df.loc[df.index == costumer_id][feature]),
                                      y1=1,
                                      x1=float(df.loc[df.index == costumer_id][feature]),
                                      line={'dash': 'dash'}),
                      row=i, col=1
                      )
        fig.update_layout(autosize=False,
                          width=750,
                          height=n * 150,
                          margin=dict(l=30, r=30, t=30, b=30))
    return fig


app = dash.Dash()
server = app.server

app.layout = html.Div([
    html.H1(children="\n\nImplémentez un modèle de scoring",
            style={'textAlign': 'center'},
            className="display-1"),

    html.H2(children="OpenClassrooms projet 7",
            style={'textAlign': 'center'},
            className="display-2"),

    html.H3(children="by Emanuele Partenza\n\n",
            style={'textAlign': 'center'},
            className="display-3"),

    html.Hr(className="my-1"),
    html.Hr(className="my-1"),

    html.H2(children="\n\nInformations personelles de votre client\n\n",
            style={'textAlign': 'center'},
            className="display-4"),

    dcc.Dropdown(id_list, None, id='first_drop'),
    dcc.Slider(
        id='slider-width', min=1, max=10,
        value=3, step=1),
    html.Div(id='first-output'),

    html.Hr(className="my-1"),
    html.Hr(className="my-1"),
    html.H2(children="\n\nInformations generales\n\n",
            style={'textAlign': 'center'},
            className="display-4"),

    html.Div(dcc.Graph(figure=px.bar(feat_importance.tail(20), x='importance', y='feature'))),
    dcc.Dropdown(df.columns, 'None', id='feature_drop'),
    dcc.Dropdown(id_list, 0, id='id_drop'),
    html.Div(id='second-output')

])


@app.callback(
    Output('first-output', 'children'),
    Input('first_drop', 'value'), Input("slider-width", "value")
)
def id_function(value, n):
    if value in id_list:
        predict_response = requests.post(url=f'https://emanuelepartenza-projet7-app.azurewebsites.net/predict',
                                         json={'sk_id': value}).json()
        features_importance_response = requests.post(
            url=f'https://emanuelepartenza-projet7-app.azurewebsites.net/features_importance',
            json={'sk_id': value}).json()

        fig_1 = predict_dash(predict_response)
        fig_2 = feature_dash(features_importance_response)
        fig_3 = px.histogram(all_proba, x='prob')
        fig_3.add_vline(x=predict_response['proba_classe_0'], line_dash='dash', line_color='firebrick')
        return (html.H3(children=f"Vous avez delectioné le  client {value}", style={'textAlign': 'center'},
                        className="display-3"),
                html.H4(children=f"Voici la probabilité que le clent {value} sera solvable :"),
                html.Div([dcc.Graph(figure=fig_1)]),
                html.Hr(className="my-2"),
                html.H4(children=f"Ici le prèmieres 20 features qui ont influencé l'algoritme dans son choix :"),
                html.Div([dcc.Graph(figure=fig_2)]),
                html.Hr(className="my-2"),
                html.H4(children=f"Vous pouvez voir le possitionement du client {value} sur la totalité des clients :"),
                html.Div([dcc.Graph(figure=fig_3)]),
                html.Div(dcc.Graph(figure=show_more_rilevant_feat(n, value))),
                )
    elif value == None:
        return html.H3(children="Selectionez l'identifiant d'un client",
                       style={'textAlign': 'center'},
                       className="display-3")
    return html.H3(children="ID is not in API",
                   style={'textAlign': 'center'},
                   className="display-3")


@app.callback(
    Output('second-output', 'children'),
    Input('feature_drop', 'value'),
    Input('id_drop', 'value'),
    prevent_initial_call=True
)
def feature_function(value, id):
    if value != 'None':
        if id != 0:
            fig_1 = px.histogram(df, x=value)
            fig_1.add_vline(x=df.loc[df.index == id, value], line_dash='dash', line_color='firebrick')
        return html.Div([dcc.Graph(figure=fig_1)])
    return


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)