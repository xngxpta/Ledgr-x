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
import pandas as pd
import datetime as dt
import plotly.express as px
import yfinance as yf
from nsepy import get_history, get_index_pe_history
import plotly.graph_objs as go
import os
import requests
import pickle

from plotly.subplots import make_subplots
from mftool import Mftool
from nsetools import Nse

from pathlib import Path
# import streamlit_authenticator as stauth
from selectolax.parser import HTMLParser

import yaml
from yaml.loader import SafeLoader
st.set_page_config(page_title='LedgrBase: Your Asset Dossier', layout="wide", initial_sidebar_state="expanded")
###---- HIDE STREAMLIT STYLE ----
#hide_st_style = """
#<style>
#    div[data-testid="metric-container"] {
#    background-color: rgba(155, 136, 225, 0.1);
#    border: 1px solid rgba(128, 131, 225, 0.1);
#    padding: 5% 5% 5% 10%;
#    border-radius: 10px;
#    color: rgb(30, 103, 119);
#    overflow-wrap: break-word;
#}
#
#/* breakline for metric text         */
#div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
#   overflow-wrap: break-word;
#   white-space: break-spaces;
#   color: yellow;
#}
#
#</style>
#    """
#st.markdown(hide_st_style, unsafe_allow_html=True)


## Authentication ##
# streamlit_app.py


## Authentication ##
##################################
direc = os.getcwd()
# st.write(direc)
# direc = f'{direc}/Documents/Ledgr'
mf = Mftool()
logofile = f'{direc}/pages/appdata/imgs/2.png'

st.sidebar.image(logofile, width=250, output_format='PNG')
start_date = dt.datetime(2021, 1, 1)
end_date = dt.datetime.today()
altstart = dt.datetime(2023, 1, 1)
indlist = pd.read_csv(f'{direc}/pages/appdata/Index_L.csv')['Symbol']
indlist = pd.Series(indlist)
list_sectoral = pd.read_csv(f'{direc}/pages/appdata/Index_L_1.csv')['ySymbol']
list_thematic = pd.read_csv(f'{direc}/pages/appdata/Index_L_2.csv')['ySymbol']
list_strat = pd.read_csv(f'{direc}/pages/appdata/Index_L_3.csv')['ySymbol']
list_fixed_income = pd.read_csv(f'{direc}/pages/appdata/Index_L_4.csv')['ySymbol']
list_bmkt = pd.read_csv(f'{direc}/pages/appdata/Index_L_0.csv')['ySymbol']
etflist = pd.read_csv(f'{direc}/pages/appdata/ETF_L.csv')['Symbol']
mflist = pd.read_csv(f'{direc}/pages/appdata/MF_L.csv')['SYMBOL']

# Pagework 1 #
hh1, hh2 = st.columns(2)
with hh1:
    st.title(": LedgrBase :")
    st.header('Your Asset Dossier')
    st.caption('Organize your asset-holdings, track their performance!')
    st.caption("**Map an Investment Portfolio right Below!!**")
with hh2:
    st.image(f'{direc}/pages/appdata/imgs/alphaA.gif', use_column_width=True)

actionsopt = ['Select Portfolio Builder','Build an Asset Portfolio',
              'Upload a Portfolio File [CSV]']

with st.container():
    # c65, c66, c67 = st.columns([1, 6, 1])
    # with c65: st.write(" ")
    # with c66: st.header('Map an Investment Portfolio Here')
    # with c67: st.write(" ")
    actions = st.selectbox('Select an Action:', actionsopt, label_visibility= "hidden")
    if actions == 'Upload a Portfolio File [CSV]':
        uploaded_file = st.file_uploader('Choose a (CSV)')
        if uploaded_file is None:
            st.stop()
        else:
            dataframe = pd.read_csv(uploaded_file)
            dataframe.to_csv(f'{direc}/pages/appdata/userinfo/uploadedpf.csv')
            st.write(dataframe.tail())
        with st.container():
            dataframe = dataframe.reset_index(drop=True)
            fig_pfup = px.line(dataframe)
            fig_pfup.update_layout(xaxis_title='Datetime', yaxis_title='')
            st.header('Asset Price Movements')
            st.plotly_chart(fig_pfup)

    elif actions == 'Build an Asset Portfolio':
        st.subheader('Create a Portfolio Here!')
        st.write('Start filling the form with relevant inputs and proceed >>')

        tickerl = pd.read_csv(f'{direc}/pages/appdata/EQUITY_L.csv')
        tickers = tickerl['SYMBOL']
        end_date = dt.datetime.today()

        with st.form('Security Ticker Info'):
            securitylist = st.multiselect('Select Assets to add:', tickers)
            start_date = st.date_input('Input Portfolio Start Date:')
            end = st.date_input('Input End Date', end_date)
            units_held = st.text_input('List purchased units seperated by a comma [eg. 12, 21, 32, ., ., ..] - ').split(', ')
            submitted = st.form_submit_button('Proceed >>')
            if submitted:
                st.success ('Processing and Plotting.....')
                units = [eval(u) for u in units_held]
                ytickers = []
                for x in securitylist:
                    y = x + '.NS'
                    # basefile = f'{direc}/pages/appdata/OHLC/{y}_data.csv'
                    ytickers.append(y)
                @st.cache_resource
                def data_get(ytickers):
                    df = yf.download(ytickers, start_date, end_date)['Close']
                    return df

                df = data_get(ytickers)

                cols = df.columns
                unitsdb = pd.DataFrame(units, index=cols)
                fig_price = px.line(df, title='Price Timelines of Selected Stocks')
                fig_price.update_layout(xaxis_title='Datetime', yaxis_title='Price Levels')

                @st.cache_resource
                def calc_pf(df, units, securitylist):
                    udx = range(len(securitylist))
                    purchase_data = df.iloc[:1, :]
                    present_data = df.iloc[-1:, :]
                    present_prices = pd.concat([present_data, purchase_data])
                    present_prices = present_prices.reset_index(drop=True)
                    present_prices = present_prices.T
                    present_prices = present_prices.reset_index()
                    present_prices = present_prices.rename(columns={'index': 'Assets',
                                                                    '0': 'Present Price',
                                                                    '1': 'Purchase Price'})
                    df3 = pd.DataFrame({'Assets': ytickers, 'Units': units},
                                       index=udx)
                    df3 = df3.merge(present_prices)
                    df3 = df3.rename(columns={0: 'Present Price', 1: 'Purchase Price'})
                    df3['Purchase Value'] = df3['Units'] * df3['Purchase Price']
                    df3['Present Value'] = df3['Units'] * df3['Present Price']
                    df3['Overall Gains'] = df3['Present Value'] - df3['Purchase Value']
                    gainratio = df3['Present Value']/df3['Purchase Value']
                    sumgains = df3['Overall Gains'].sum().round(2)
                    invested_amount = df3['Purchase Value'].sum().round(1)
                    totalgainratio = 100*sumgains/invested_amount
                    cumgains = round(sumgains, 2)
                    sumunits = df3['Units'].sum()
                    sumwealth = df3['Present Value'].sum().round(2)
                    figPortfolioPie = px.pie(df3, values=df3['Units'], hole=0.5, names=df3['Assets'])
                    figPortfolioPie.update_traces(textposition='auto', textinfo='percent+label+value+text')
                    figPortfolioPie.update_layout(annotations=[dict(text=f'Units Owned:{sumunits}',
                                                                    x=0.5, y=0.5, font_size=12, showarrow=False)])
                    figPortfolioPie.update_layout(showlegend=False)
                    figPortfolioWealth_present = px.pie(df3, values=df3['Present Value'], hole=0.5, names=df3['Assets'])
                    figPortfolioWealth_present.update_traces(textposition='outside', textinfo='value+label+text')
                    figPortfolioWealth_present.update_layout(annotations=[dict(text=f'Total Value: \n{sumwealth}', x=0.5, y=0.5, font_size=14,
                                                                               showarrow=False)])
                    figPortfolioProfits_present = px.bar(df3, x=df3['Assets'], y=df3['Overall Gains'], color=df3['Assets'])

                    figPortfolioProfits_present.update_layout(showlegend=False)
                    figPortfolioWealth_present.update_layout(showlegend=False)
                    return df3, gainratio,totalgainratio, sumgains, sumwealth, invested_amount, sumunits, figPortfolioProfits_present, figPortfolioPie, figPortfolioWealth_present, cumgains, invested_amount

                df3, gainratio,totalgainratio, sumgains, sumwealth, invested_amount, sumunits, figPortfolioProfits_present, figPortfolioPie, figPortfolioWealth_present, cumgains, invested_amount = calc_pf(df, units, securitylist)

                cb1, cb2, cb3 = st.columns([1, 8, 1])
                with cb1: st.write(" ")
                with cb2:
                    st.subheader('Portfolio Holdings Summary')
                    df3 = df3.set_index(["Assets"], drop=True)
                    st.write(df3)
                with cb3: st.write(" ")
                # st.write(df3)
                st.plotly_chart(fig_price, use_container_width=True)
                with st.expander("Access Portfolio Data here!!"):
                    st.write("Portfolio Data", df)
                colx111, colx112, colx113, colx114 = st.columns(4)
                with colx111:  st.metric('Cumulative Portfolio Value', sumwealth.round(2))
                with  colx112: st.metric("Total Profits or Losses", sumgains.round(2))
                with colx113: st.metric('Profit Percentage', totalgainratio.round(2))
                with colx114: st.metric("Invested Value", invested_amount)
                c1, c2= st.columns([2, 3])
                with c1:
                    st.subheader('Asset Holdings in Portfolio')
                    st.plotly_chart(figPortfolioPie, use_container_width=True)
                with c2:
                    st.subheader('Profit Map')
                    st.plotly_chart(figPortfolioProfits_present, use_container_width=True)

                st.subheader('Portfolio Wealth Map')
                st.plotly_chart(figPortfolioWealth_present)
                pass


st.write("  --------------  ")

@st.cache_resource
def data_NSEI():
    df_NSEI = yf.download("^NSEI", period='250d', interval='1d')
    # df_NSEI.to_csv(f'{direc}/pages/appdata/IndexData/NSEI_data.csv')

    figOHLC_NSEI = go.Figure()
    figOHLC_NSEI.add_trace(go.Ohlc(x=df_NSEI.index, open=df_NSEI["Open"], high=df_NSEI["High"],
        low=df_NSEI["Low"], close=df_NSEI["Close"],
        name="NSEI"))
    figOHLC_NSEI.update_layout(xaxis_rangeslider_visible=False)
    return df_NSEI, figOHLC_NSEI

df_NSEI, figOHLC_NSEI = data_NSEI()

@st.cache_resource
def data_SPX():
    df_SPX = yf.download("^SPX", period='250d', interval='1d')
    # df_SPX.to_csv(f'{direc}/pages/appdata/IndexData/SPX_data.csv')
    figOHLC_SPX = go.Figure()
    figOHLC_SPX.add_trace(go.Ohlc(x=df_SPX.index, open=df_SPX["Open"], high=df_SPX["High"],
    low=df_SPX["Low"], close=df_SPX["Close"],
    name="SPX"))
    figOHLC_SPX.update_layout(xaxis_rangeslider_visible=False)
    return df_SPX, figOHLC_SPX

df_SPX, figOHLC_SPX = data_SPX()

@st.cache_resource
def data_DAX():
    df_DAX = yf.download("^GDAXI", period='250d', interval='1d')
    figOHLC_DAX = go.Figure()
    figOHLC_DAX.add_trace(go.Ohlc(x=df_NSEI.index, open=df_DAX["Open"], high=df_DAX["High"],
    low=df_DAX["Low"], close=df_DAX["Close"],
    name="DAX"))
    figOHLC_DAX.update_layout(xaxis_rangeslider_visible=False)
    return df_DAX, figOHLC_DAX

df_DAX, figOHLC_DAX = data_DAX()

@st.cache_resource
def data_CAC():
    df_CAC = yf.download("^FCHI", period='250d', interval='1d')
    # df_CAC.to_csv(f'{direc}/pages/appdata/IndexData/CAC40_data.csv')

    figOHLC_CAC = go.Figure()
    figOHLC_CAC.add_trace(go.Ohlc(x=df_CAC.index, open=df_CAC["Open"], high=df_CAC["High"],
    low=df_CAC["Low"], close=df_CAC["Close"],
    name="CAC40"))
    figOHLC_CAC.update_layout(xaxis_rangeslider_visible=False)
    return df_CAC, figOHLC_CAC

df_CAC, figOHLC_CAC = data_CAC()

@st.cache_resource
def data_DJIA():
    df_DJIA = yf.download("DJIA", period='250d', interval='1d')
    # df_DJIA.to_csv(f'{direc}/pages/appdata/IndexData/DJIA_data.csv')
    figOHLC_DJIA = go.Figure()
    figOHLC_DJIA.add_trace(go.Ohlc(x=df_DJIA.index, open=df_DJIA["Open"], high=df_DJIA["High"],
    low=df_DJIA["Low"], close=df_DJIA["Close"],
    name="DJIA"))
    figOHLC_DJIA.update_layout(xaxis_rangeslider_visible=False)
    return df_DJIA, figOHLC_DJIA

df_DJIA, figOHLC_DJIA = data_DJIA()

@st.cache_resource
def data_TYO():
    df_TYO = yf.download("^N225", period='250d', interval='1d')
    # df_TYO.to_csv(f'{direc}/pages/appdata/IndexData/N225_data.csv')
    figOHLC_TYO = go.Figure()
    figOHLC_TYO.add_trace(go.Ohlc(x=df_TYO.index, open=df_TYO["Open"], high=df_TYO["High"],
    low=df_TYO["Low"], close=df_TYO["Close"],
    name="DJIA"))
    figOHLC_TYO.update_layout(xaxis_rangeslider_visible=False)
    return df_TYO, figOHLC_TYO
df_TYO, figOHLC_TYO = data_TYO()


@st.cache_resource
def data_FTSE():
    df_FTSE = yf.download("^FTSE", period='250d', interval='1d')
    # df_FTSE.to_csv(f'{direc}/pages/appdata/IndexData/FTSE_data.csv')
    figOHLC_FTSE = go.Figure()
    figOHLC_FTSE.add_trace(go.Ohlc(x=df_FTSE.index, open=df_FTSE["Open"], high=df_FTSE["High"],
    low=df_FTSE["Low"], close=df_FTSE["Close"],
    name="FTSE"))
    figOHLC_FTSE.update_layout(xaxis_rangeslider_visible=False)
    return df_FTSE, figOHLC_FTSE

df_FTSE, figOHLC_FTSE = data_FTSE()

@st.cache_resource
def data_mkt():
    df_mk = pd.DataFrame()
    df_mk['NSEI'] = df_NSEI['Close']
    df_mk['DAX'] = df_DAX['Close']
    df_mk['CAC'] = df_CAC['Close']
    df_mk['SPX'] = df_SPX['Close']
    df_mk['FTSE'] = df_FTSE['Close']
    df_mk['N225'] = df_TYO['Close']
    fig_mkt = go.Figure()
    fig_mkt.add_trace(go.Scatter(x= df_NSEI.index, y=df_NSEI['Close'],
                        mode='lines',
                        name='NSEI'))
    fig_mkt.add_trace(go.Scatter(x= df_DAX.index, y=df_DAX['Close'],
                        mode='lines',
                        name='DAX'))
    fig_mkt.add_trace(go.Scatter(x= df_CAC.index,y=df_CAC['Close'],
                        mode='lines', name='CAC'))
    fig_mkt.add_trace(go.Scatter(x= df_SPX.index,y=df_SPX['Close'],
                        mode='lines', name='SPX'))
    fig_mkt.add_trace(go.Scatter(x= df_DJIA.index,y=df_DJIA['Close'],
                        mode='lines',
                        name='DJIA'))
    fig_mkt.add_trace(go.Scatter(x= df_FTSE.index,y=df_FTSE['Close'],
                        mode='lines',
                        name='FTSE'))
    fig_mkt.add_trace(go.Scatter(x= df_TYO.index, y=df_TYO['Close'],
                        mode='lines',
                        name='N225'))
    # fig_mkt.update_layout(legend=dict(
    #     orientation="h",
    #     entrywidth=30,
    #     yanchor="bottom",
    #     y=1,
    #     xanchor="center",
    #     x=1
    # ))

    fig_mkt.update_xaxes(visible=True, showticklabels=True)
    fig_mkt.update_yaxes(visible=True, showticklabels=True)

    return df_mk, fig_mkt

df_mk, fig_mkt = data_mkt()
multi_symbols = ['^IXIC' , '^GSPC', '^NYA', '^BSESN' , '^NSEI', '^NSEBANK']
multi_details = ['NASDAQ Composite', 'S&P500', 'NYSE Composite (DJ)', 'BSE SENSEX',  'NIFTY50','NIFTYBANK']

multi_index_list = pd.DataFrame({"Symbol" : multi_symbols, "Exchange Index" : multi_details})

# ^TYX - US Treasury Yield - 30 years
@st.cache_resource
def treasury():
    df_treasury = yf.download('^TYX', period = '3y')
    fig_treasury = go.Figure()
    fig_treasury.add_trace(go.Ohlc(x=df_treasury.index, open=df_treasury["Open"],
        high=df_treasury["High"],
        low=df_treasury["Low"],
        close=df_treasury["Close"]))
    fig_treasury.update_xaxes(visible=True, showticklabels=True)
    fig_treasury.update_yaxes(title = 'US Treasury Yield', visible=True, showticklabels=True)
    fig_treasury.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
    return df_treasury, fig_treasury

df_treasury, fig_treasury = treasury()

@st.cache_resource
def vix():
    df_vix = yf.download('^VIX', period = '2y')
    # df_vix = df_vix.drop(['Volume'], axis=1)
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Ohlc(x=df_vix.index, open=df_vix["Open"], high=df_vix["High"],
        low=df_vix["Low"], close=df_vix["Close"],
        name="VIX"))
    fig_vix.update_traces(increasing_line_color= 'yellow', decreasing_line_color= 'gray')
    fig_vix.update_layout(xaxis_rangeslider_visible=False)
    fig_vix.update_xaxes(visible=True, showticklabels=True)
    fig_vix.update_yaxes(title = 'VIX', visible=True, showticklabels=True)
    fig_vix.update_layout(height = 360, showlegend=False)
    return  df_vix, fig_vix

df_vix, fig_vix = vix()

hg1, hg2 = st.columns(2)
with hg1:
    st.title(': MarketBoard :')
    st.subheader('Follow, Track and Global Markets')
    st.caption('Explore Indices, Exchange Traded & Mutual Funds and more!')
with hg2:
    st.image(f'{direc}/pages/appdata/imgs/Marketbase-Anim.gif', use_column_width=True)

with st.container():
    st.subheader('A. Markets & Exchanges')

    tub0, tub1, tub2, tub3, tub4, tub5, tub6, tub7 = st.tabs(['Global Markets', "NSE - IN", "SPX - USA", "DAX - GDR", 'CAC40 - FR', 'Dow Jones - USA', 'Nikkei 225 - JPN', "FTSE - UK"])
    with tub0:
        st.plotly_chart(fig_mkt, use_container_width=True)
    with tub1:
        st.plotly_chart(figOHLC_NSEI, use_container_width=True)
    with tub2:
        st.plotly_chart(figOHLC_SPX, use_container_width=True)
    with tub3:
        st.plotly_chart(figOHLC_DAX, use_container_width=True)
    with tub4:
        st.plotly_chart(figOHLC_CAC, use_container_width=True)
    with tub5:
        st.plotly_chart(figOHLC_DJIA, use_container_width=True)
    with tub6:
        st.plotly_chart(figOHLC_TYO, use_container_width=True)
    with tub7:
        st.plotly_chart(figOHLC_FTSE, use_container_width=True)

st.write("  --------  ")##

with st.container():
    ms1, ms2, ms3 = st.columns([4, 1, 4])
    with ms1:
        st.subheader("B. SIP Calculator")
        st.caption("Find out your Returns from any SIP scheme.")
        # with st.form('sipcalc'):
        A = st.slider("Enter the monthly SIP amount: ", min_value=500, max_value=99900, value=1250, step=200, help="Input your monthly payments installments here!")
        YR = st.slider("Enter the yearly Rate of Return in pct: ", min_value=5, max_value=25, value=10, step=1, help="Indicate your scheme's Return Rate as mentioned in its documentation or a close ball-park [ref: IRR or XIRR]")
        Y = st.slider("Enter the number of years: ", min_value=2, max_value=25, value=3, step=1, help="Indicate the nmber of years which you have been investing into this scheme for by moving the slider")
        # submitted  = st.form_submit_button("Calculate Returns >> ")
        # if submitted:
        MR = YR/12/100
        M = Y * 12
        FV = A * ((((1 + MR)**(M))-1) * (1 + MR))/MR
        FV = round(FV)
        gh2, gh3 = st.columns(2)
        with gh2: st.subheader("Your Expected Returns are: - ")
        with gh3: st.metric("Returns [INR]",FV)
    with ms2:
        st.write("  ")
    with ms3:
        st.subheader('C. Exchange Traded & Index Funds')
        #with st.form('etfdata'):
        etfselect = st.selectbox("Please select ETF here!", etflist)
        # submitted = st.form_submit_button("Submit!!")
            # if submitted:
        @st.cache_resource
        def etf(etfselect):
            etfselect = etfselect + '.NS'
            df_etf =yf.download(etfselect)
            figOHLC_etf= make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2,
                                subplot_titles=('NAV Price Movement', 'Traded Volume'),
                                row_width=[0.2, 0.7])
            figOHLC_etf.add_trace(go.Ohlc(x=df_etf.index, open=df_etf["Open"], high=df_etf["High"],
                                             low=df_etf["Low"], close=df_etf["Close"],
                                             name=f"OHLC for {etfselect}"), row=1, col=1)
            figOHLC_etf.add_trace(go.Bar(
                x=df_etf.index, y=df_etf['Volume'], name='Volume Traded', showlegend=False), row=2, col=1)
            figOHLC_etf.update_layout(xaxis_rangeslider_visible=False)
            figOHLC_etf.update_layout(showlegend=False)
            return figOHLC_etf, df_etf

        figOHLC_etf, df_etf = etf(etfselect)
        st.plotly_chart(figOHLC_etf, use_container_width=True)

# mflist2 = mf.get_scheme_codes()
with st.container():
    mm1, mm2 = st.columns(2)
    with mm1:
        st.subheader("D. Mutual Funds")
        st.caption("**Get current quotes & comprehensive information on any Mutual Fund.**")
    mflist2 = mf.get_scheme_codes()
    mfdb = pd.DataFrame(mflist2.items())
  #  mfdb.to_csv("MFlist3.csv")
    mfdb2 =pd.DataFrame({'Fund Description' : mfdb[1].values}, index = mfdb[0])
    with mm2:
        st.caption("Find your Mutual 'Fund Code' listed in the expander here!")
        with st.expander("Get the Fund Codes and Names Here!!"):
            st.write(mfdb2)
    with st.form("mfinfo"):
        mfselect = st.selectbox('Select Mutual Fund by fund code: ', mfdb2.index)
        submitted = st.form_submit_button("Submit!")
        if submitted:
            mfselect = int(mfselect)
            mf_final_b = mfdb2[f'{mfselect}':]
            mf_final = mf_final_b['Fund Description'].iloc[0]
            mfdb23 = mfdb2['Fund Description']

            fs1, fs2, fs3 = st.columns([1, 6, 1])
            with fs1:
                st.subheader(" ")
            with fs2:
                st.write("Fund Selected: ", mf_final)
            with fs3:
                st.subheader(" ")
            st.write(" ---- ")
            @st.cache_resource
            def mf_calc(mfselect):
                #schemes = mf.get_available_schemes(f'{mfselect}')
                #available_schemes = pd.DataFrame(schemes.items())
                mfquote = mf.get_scheme_quote(mfselect)
                mfdetails = mf.get_scheme_details(mfselect)
                #mfinfo_more = mf.get_scheme_info(mfselect)
                mfdata = mf.get_scheme_historical_nav(mfselect,as_Dataframe=True)
                return mfquote, mfdetails, mfdata

            mfquote, mfdetails,  mfdata = mf_calc(mfselect)

            nav1 = mfdetails['scheme_start_date']['nav']
            nav1 = float(nav1)
            nav2 = mfquote['nav']
            nav2 = float(nav2)
            nav_delta = nav2 - nav1
            nav_delta_pct1 = nav_delta/nav1
            fig_nav = px.line(mfdata['nav'])
            fig_daychng = px.bar(x=mfdata.index, y=mfdata['dayChange'])
            # st.metric("Instrument Name", mfdetails["scheme_name"])
            md1, md2, md3, md4 = st.columns(4)
            with md1: st.metric("Present NAV", round(nav2, 2))
            with md2: st.metric("NAV Gain/Losses", round(nav_delta, 2))
            with md3: st.metric('Initial NAV', round(nav1, 2))
            with md4: st.metric("NAV Del-Ratio", round(nav_delta_pct1, 2))
            mf1, mf2, mf3 = st.columns(3)
            with mf1:
                st.metric("Scheme code", mfdetails["scheme_code"])
            with mf2:
                st.metric('Scheme Start Date', mfdetails['scheme_start_date']['date'])
            with mf3:
                st.metric("Scheme Category", mfdetails["scheme_category"])
            me1, me2 = st.columns(2)
            with me1: st.metric("Fund House", mfdetails['fund_house'])
            with me2: st.metric("Instrument Type", mfdetails['scheme_type'])

            with st.expander("Get collated MF Data here!!"):
                st.write("Fund Quotation", mfquote)
                st.write("Details", mfdetails)
            st.plotly_chart(fig_nav)
            st.plotly_chart(fig_daychng)
st.write("  --------  ")## E
currency_list = pd.read_csv(f'{direc}/pages/appdata/currency_list.csv')

@st.cache_resource
def currency(currency_selected):
    currency_df = yf.download(currency_selected, period='1y')
  #  currency_df.to_csv(f'{direc}/pages/appdata/OHLC/{currency_selected}.csv')
    currency_df1 = currency_df.filter(['Open', 'High', 'Low', 'Close'], axis=1)
    fig_currency1 = go.Figure()
    fig_currency1.add_trace(go.Ohlc(x=currency_df1.index, open=currency_df1["Open"],
                                    high=currency_df1["High"],
                                    low=currency_df1["Low"],
                                    close=currency_df1["Close"]))
    fig_currency1.update_xaxes(visible=True, showticklabels=True)
    fig_currency1.update_yaxes(title = 'Exchange Ratio', visible=True, showticklabels=True)
    fig_currency1.update_layout(xaxis_rangeslider_visible=False, height=360)
    return currency_df1, fig_currency1


@st.cache_resource
def riskFreeInterestRate(url: str = "https://www.rbi.org.in/",) -> None:
    response = HTMLParser(requests.get(url).content)
    selector = "#wrapper > div:nth-child(10) > table"
    data = [node.html for node in response.css(selector)]
    df_rbi = (pd.read_html(data[0])[0][4:13])
    df_rbi.columns = ["GovernmentSecurityName", "Percent"]
    df_rbi.reset_index(inplace=True,drop=True)
    df_rbi["GovernmentSecurityName"] = df_rbi["GovernmentSecurityName"].str.rstrip(' ').str.lstrip(' ')
    df_rbi["Percent"] = df_rbi["Percent"].str.rstrip('% #').str.rstrip('%*').str.lstrip(':  ')
    df_rbi = df_rbi.astype({'GovernmentSecurityName': 'str', 'Percent': 'float32'}, copy=False)
    with open("RiskFreeInterestRate.json", "w") as jsonFile:
        jsonFile.write(df_rbi.to_json(orient='records'))
    return df_rbi, jsonFile

df_rbi, jsonFile = riskFreeInterestRate()

# df_rbi.to_csv[f'{direc}/pages/appdata/IndexData/RBI_{end_date}.csv']
rbi_yield =  df_rbi['Percent'].iloc[-1]
figRBI = px.line(df_rbi['Percent'], labels=df_rbi['GovernmentSecurityName'])
figRBI.update_xaxes(visible=False, showticklabels=False)
figRBI.update_yaxes(title = 'Periodic Rates', visible=True, showticklabels=True)
figRBI.update_layout(title = 'RBI Yield Rate', height = 360, showlegend=False)
with st.container():
    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("E. Foreign Exchange & Currencies")
    with cc2:
         currency_selected = st.selectbox("Choose Currency Pair > ", currency_list)
    currency_df1, fig_currency1= currency(currency_selected)
    st.plotly_chart(fig_currency1, use_container_width=True)
    l_curr = currency_df1.iloc[-1]
    lcv0, lcv1, lcv2, lcv3, lcv4 = st.columns(5)
    with lcv0: st.subheader("*Latest Exchange Data*")
    with lcv1: st.metric('Open', l_curr['Open'].round(2))
    with lcv2: st.metric('High', l_curr['High'].round(2))
    with lcv3: st.metric('Low', l_curr['Low'].round(2))
    with lcv4: st.metric('Close', l_curr['Close'].round(2))

st.write("  --------  ")###
with st.container():
    df_vix, fig_vix = vix()
    l_vix = df_vix.iloc[-1]
    mv1, mv2 = st.columns(2)
    with mv1:
        st.header("F. Market Volatility Index")
    with mv2:
        st.metric("Market VIX", l_vix['Close'].round(2))
    st.plotly_chart(fig_vix, use_container_width=True)

    df_treasury, fig_treasury = treasury()
    l_ustreasury = df_treasury['Close'].iloc[-1]
    l_rbi = df_rbi["Percent"].iloc[-1]
    l_rbi = l_rbi.round(2)
    bn1, bn2, bn3 = st.columns([2, 1, 1])
    with bn1:
        st.header("G. Treasury Yield Rates")
        st.caption('''Estimate the real risk-free rate >> Yield of the Treasury bond - Inflation Rate''')
    with bn2: st.metric("US Treasury", l_ustreasury.round(2))
    with bn3: st.metric("RBI", round(l_rbi, 2))

    tr1, tr2 = st.tabs(["US Treasury", "Reserve Bank of India"])
    with tr1: st.plotly_chart(fig_treasury, use_container_width=True)
    with tr2: st.plotly_chart(figRBI, use_container_width=True)

st.write("  --------  ")
ft1, ft2, ft3 = st.columns([1, 5, 1])
with ft1: st.write(" ")
with ft2: st.caption(": | 2023 - 2024 | All Rights Resrved  ©  Ledgr Inc. | www.alphaLedgr.com | alphaLedgr Technologies Ltd. :")
with ft3: st.write(" ")
