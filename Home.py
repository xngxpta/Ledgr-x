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

import streamlit as st
import os
import streamlit as st
#from st_functions import st.image, load_css

direc = os.getcwd()
#link = '[GitHub](http://github.com)'
icon_size = 100
logofile = f'{direc}/pages/appdata/imgs/2.png'
st.markdown(''' <div align="center"><h1>Hello! Welcome to Ledgr.</h1></div>''', unsafe_allow_html=True)
st.markdown(''' <div align="center"><h3>Learn how to get started on the platform!</h3></div>''', unsafe_allow_html=True)
st.markdown(''' <div align="center"><h3>See below for details</h3></div>''', unsafe_allow_html=True)
with st.container():
    a1,a2 = st.columns([1, 4])
    with a1:
        st.image(f'{direc}/pages/appdata/imgs/LedgrBase.png', 'LedgrBase', icon_size)
    with a2:
        st.subheader("[Ledgrbase](http://localhost:8501/LedgrBase)")
        st.write("Map your existing asset holdings and portfolios. Review and note their overall performance till date.")
st.write("----------------------")
with st.container():
    b1, b2 = st.columns([4, 1])
    with b1:
        st.subheader("[MarketBoard](http://localhost:8501/LedgrBase#marketboard)")
        st.write("Map your existing asset holdings and portfolios. Review and note their overall performance till date.")
    with b2:
        st.image(f'{direc}/pages/appdata/imgs/MarketBoard.png', 'MarketBoard', icon_size)
st.write("----------------------")
with st.container():
    c1, c2 = st.columns([1, 4])
    with c1:
        st.image(f'{direc}/pages/appdata/imgs/AnalyticsBox.png', 'AnalyticsBox', icon_size)
    with c2:
        st.subheader("[AnalyticsBox](http://localhost:8501/AnalyticsBox)")
        st.write("Analyze a Security In-Depth. Visualize Metrics & Key Indicators.")
        st.write("Note Signals and Indicators. Gather comprehensive knowhow on a selected Security.")
st.write("----------------------")
with st.container():
    d1, d2 = st.columns([4, 1])
    with d1:
        st.subheader("[InvestmentLab](http://localhost:8501/InvestmentLab)")
        st.write("Optimize Investment Allocations. Generate Efficient Investment Portfolios to Maximize Returns.")
        st.write("Input assets and amount. Select any from 5 Optimized portfolios presented.")
    with d2:
        st.image(f'{direc}/pages/appdata/imgs/InvestmentLab.png', 'InvestmentLab', icon_size)
st.write("----------------------")
with st.container():
    e1, e2 = st.columns([1, 4])
    with e1:
        st.image(f'{direc}/pages/appdata/imgs/ForecastEngine.png', 'ForecastEngine', icon_size)
    with e2:
        st.subheader("[ForecastEngine](http://localhost:8501/ForecastEngine)")
        st.write("Train Ledgr's AI and generate price forecasts for any securities or currencies.")
        st.write("Observe overall trend plots aover multiple timescales")
st.write("----------------------")
