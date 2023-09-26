#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:57:06 2023

@author: r_xn
"""
# !/usr/bin/env python310
# -*- coding: utf-8 -*-
# __author__ = 'R. Sengupta | r_xn'
# __copyright__ = 'Copyright 2023, Ledgr | www.alphaLedgr.com'
# __credits__ = ['r_xn, s.sengupta, prithvirajsengupta033@gmail.com, adasgupta@gmail.com']
# __license__ = 'Ledgr | alphaledgr.com'
# __version__ = '01.02.04'
# __maintainer__ = 'r_xn@alphaledgr.com'
# __emails__ = 'r_xn@alphaledgr.com / response@alphaledgr.com'
# __status__ = 'In active development'

# Imports

import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import matplotlib as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import os
import streamlit as st
bpath = os.getcwd()
# direc = f'{direc}/Documents/Ledgr'
st.set_page_config(page_title='Ledgr: Forecasting Engine', layout="wide", initial_sidebar_state="expanded")

#st.markdown("""
#    <style>
#        div[data-testid="metric-container"] {
#        background-color: rgba(155, 136, 225, 0.1);
#        border: 1px solid rgba(128, 131, 225, 0.1);
#        padding: 5% 5% 5% 10%;
#        border-radius: 10px;
#        color: rgb(30, 103, 119);
#        overflow-wrap: break-word;
#        }
#
#        /* breakline for metric text         */
#        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
#        overflow-wrap: break-word;
#        white-space: break-spaces;
#        color: yellow;
#        }
#    </style>
#    """, unsafe_allow_html=True)
bpath = f'{bpath}/pages'
logofile = f"{bpath}/appdata/imgs/2.png"
# #########################################
start_date = dt.datetime(2020, 1, 1)
end_date = dt.datetime.today()
pathtkr = f"{bpath}/appdata/EQUITY_L.csv"
vidfile = f"{bpath}/appdata/imgs/Forecast-Anime.gif"


tickerdb = pd.read_csv(pathtkr)
tickerlist = tickerdb["SYMBOL"]

with st.sidebar:
    st.image(logofile, width=250)
fc1, fc2 = st.columns(2)
with fc1:
    st.title(": Forecast Engine :")
    st.caption("Train Ledgr's AI Engines. Forecast Asset Prices.")
    st.info("Chart behaviour, predict price-ranges, observe trajectories.")
with fc2:
    st.image(vidfile, use_column_width=True)

with st.form("pfinputs"):
    st.subheader("Inputs for the Forecasting Engine")
    stock = st.selectbox("Please Select a Security Symbol", tickerlist)
    submitted = st.form_submit_button("Submit")
    if not submitted:
        st.error("Please input details and click 'Submit' !!")
        st.stop()
    if submitted:
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Thanks! Forecasting Commences!")
        @st.cache_data
        def getdata(stock):
            stock = stock + ".NS"
            df = yf.download(stock, period = '800d', interval='1d')
            return df

        df = getdata(stock)
        pass

df = getdata(stock)

open =  df['Open']
hi =  df['High']
lo =  df['Low']
close = df['Close']
prof_df_close = pd.DataFrame({"ds" : df.index, "y": df['Close']})
prof_df_close = prof_df_close.reset_index()
m = Prophet()
m.fit(prof_df_close)
future_year = m.make_future_dataframe(periods=150)
forecast_year = m.predict(future_year)
m.plot(forecast_year)
m.plot_components(forecast_year)
a = plot_plotly(m, forecast_year)
a.update_xaxes(title="Timeline", visible=True, showticklabels=True)
a.update_yaxes(title="Predicted Prices (INR)", visible=True, showticklabels=True)
b = plot_components_plotly(m, forecast_year)
b.update_xaxes(title="Timeline", visible=True, showticklabels=True)
b.update_yaxes(title="Predicted Prices (INR)", visible=True, showticklabels=True)
dx = forecast_year.filter(["ds", 'yhat'], axis=1)
dx = dx.set_index(['ds'])
dx.rename(columns={'yhat': 'Predictions'}, inplace=True)
c = px.line(dx)
c.add_trace(go.Scatter(x=dx.index, y=df['Close'], name='Close'))
c.update_xaxes(title='Timeline', showticklabels=True, visible=True)
c.update_yaxes(title="Price Data", showticklabels=True, visible=True)
c.update_layout(legend=dict(
    orientation="h",
    entrywidth=100,
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ))

k1,k2, k3 = st.columns([3, 5, 3])
with k1: st.write(" ")
with k2: st.subheader(f"{stock} Forecast Plot")
with k3: st.write(" ")
st.plotly_chart(a, use_container_width=True)
with st.expander("Get Forecast Data Here!"):
    st.write(forecast_year.iloc[-150:])
    st.write(forecast_year.iloc[-100:])
st.plotly_chart(c, use_container_width=True)
j1,j2, j3 = st.columns([3, 5, 3])
with j1: st.write(" ")
with j2: st.subheader(f"{stock} Price Trajectory")
with j3: st.write(" ")
st.plotly_chart(b, use_container_width=True)
st.write("  -----------  ")

ft1, ft2, ft3 = st.columns([1, 5, 1])
with ft1: st.write(" ")
with ft2: st.caption(": | 2023 - 2024 | All Rights Resrved  ©  Ledgr Inc. | www.alphaLedgr.com | alphaLedgr Technologies Ltd. :")
with ft3: st.write(" ")
