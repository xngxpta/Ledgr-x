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

import pandas as pd
import streamlit as st
import base64
import os
st.set_page_config(page_title='About', layout="wide", initial_sidebar_state="expanded")

#
#st.markdown("""
#<style>
#div[data-testid="metric-container"] {
#   background-color: rgba(155, 136, 225, 0.1);
#   border: 1px solid rgba(128, 131, 225, 0.1);
#   padding: 5% 5% 5% 10%;
#   border-radius: 10px;
#   color: rgb(30, 103, 119);
#   overflow-wrap: break-word;
#}
#
#/* breakline for metric text         */
#div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
#   overflow-wrap: break-word;
#   white-space: break-spaces;
#   color: yellow;
#}
#</style>
#"""
#, unsafe_allow_html=True)
direc = os.getcwd()
# direc = f'{direc}/Documents/Ledgr'
logofile = f"{direc}/pages/appdata/imgs/2.png"

st.sidebar.image(logofile, width=250, output_format='PNG')


n1, n2, n3 = st.columns([2, 6, 2])
with n1: st.write(" ")
with n2:
    st.title("About & Documentation")
with n3: st.write(" ")
n12, n13, n14 = st.columns([2, 4, 2])
with n12: st.write(" ")
with n13:
    st.header("How does Ledgr work")
with n14: st.write(" ")
st.write("  ------  ")
m1, m2 = st. columns(2)
with m1:
    st.subheader("Navigate using the Sidebar!!")
    st.markdown(
    """
    - Each app has its unique functionality and presents unique information to the user.

    - Find out more by clicking on them!! Choose any via the Sidebar > .
    - On each page -

        (a) input relevant details,

        (b) authorize the access fee payment

        (c) click submit!!. That's it.

    """)
with m2:
    st.subheader("Each module is a vending machine!!")
    st.markdown(
    """
    - Assess information, data, plots and signals.

    - Gather overviews as well as in-depth analyses on various asset-classes and the market.

    - Get a comprehensive picture of your total wealth, along with in-depth insights as you navigate through the app-engines.

    - Use LedgrTokens to activate engines and modules as you would like. Analyze, assess and access the data as a download file.

    """)

st.write("  ------  ")
lb1,lb2 = st.columns([2, 5])
with lb1:
    st.subheader(":LedgrBase:")
with lb2:
    st.subheader("Map your investment portfolios here!")
st.markdown(
"""
- Organize all your expenses at the LedgrBase, which by default is the homepage as one Signs In.
- Unify your asset holdings here. Visualize everything in a set of convenient, interactive dashboards.

- Build a new Portfolio [For new Users], save it to your profile or attach it to your dossier.

- Track each portfolio at the LedgrBase.

- Add other asset holdings to 'Your Locker'. Asset classes accepted are Securities, Derivatives, Bonds, ETFs, MFs, Crypto, NFTs & Fiat Money. [Provisions for other classes shall be included shortly]

- Trace Performance, gauge Returns due and gain a clear overview of your Wealth at LedgrBase
""")
mb1, mb2 = st.columns([2, 5])
with mb1: st.subheader(":MarketBoard:")
with mb2: st.subheader("Follow Markets, Explore Funds & SIPs and more... ")
vv23, vv33 = st.columns([1, 4])
with vv23:
    st.image(f"{direc}/pages/appdata/imgs/MarketBoard1.png")
with vv33:
    st.markdown(
    """
- Follow Markets, trace Market Indices.
- Track their performances over time along with other Markets & Exchanges.
- Visualize their comparative performance on the opening chart, navigate through the tabs for specific economies/indices.

- Explore ETFs and MFs. Get quotes & information summaries.

- Calculate SIP Returns on the SIP Calculator, find out your next Returns by inputting relevant data.

- Know about Performers, Sectoral Activities and other commodities or currencies like BTC, ETH or EUR-USD.

- Get info on Derivatives like Futures and Option Chains.

- Get exchange rates and track Volatility indices like India VIX.

    """)
st.write("  ------  ")

ant1, ant2 = st.columns([2, 5])
with ant1: st.subheader(":AnalyticsBox:")
with ant2: st.subheader("Explore Securities In Depth, Analyze KPIs and Signals!!")
va23, va33 = st.columns([1, 4])
with va23:
    st.image(f"{direc}/pages/appdata/imgs/Analytics1.png")
with va33:
    st.markdown(
    """

    - Get OHLC Price Charts, Volume plots and all relevant information on any security.

    - Access 42+ technical indicators, peruse stochastic signals.

    - Perform complex analyses on Securities based on real-time and historical data.

        - Generate data reports and download calculated data at will.

        - Follow Price Movements and other KPIs.

    -  Gain insights essential to make informed decisions and ensure maximum returns for your trades or investments.

    """)
st.write("  ------  ")

im1, iml = st.columns([2, 5])
with im1: st.subheader(":InvestmentLab:")
with iml:
    st.subheader("Optimize Portfolios, generate efficient allocations")
    st.caption("Enjoy greater Returns!")

vg23, vg33 = st.columns([1, 4])
with vg23:
    st.image(f"{direc}/pages/appdata/imgs/InvestLab1.png")
with vg33:st.markdown("""
    - Optimize your Investments! Start with a new perspective.
    - Input Securities to include in your Portfolio.
    - Indicate the total amount that you would be willing to allocate !!

    - Drop in the access fees and click submit !!

    *Users with Cashcards have their token automatically deducted from their connected LedgrWallets.*

    - InvestmentLab presents 5 different sets of optimally allocated portfolios with expected outcomes as per your inputs.

    - Choose any one and invest as indicated to Generate Maximum Returns using the InvestmentLab.

    - Alternatively, one may select an allocation which Minimizes the Risk Exposure.

    - Or, try out many other combinations of stocks and see how much more gains are possible!!""")

st.write("  ------  ")

fe1, fe2 = st.columns([2, 5])
with fe1: st.subheader(":ForecastEngine:")
with fe2: st.subheader("Train ML-AI Engines, Predict Prices and gather intelligence")
vv23, vv33 = st.columns([1, 4])
with vv23:
    st.image(f"{direc}/pages/appdata/imgs/Forecast1.png")
with vv33:
    st.markdown(
    """
    - Predict future price points for any asset or security with Ledgr's AI ForecastingEngines.
    - Get Price Forecasts & Sentimental Analyses on specific securites, overall Markets or specific Market Segments!!

    - Assess yearly, monthly as well as weekly motion of the prices.
    - Explore how security prices move through selected timespan.
    - Run it yourself, by your own rules. Use Ledgr's LSTM, ARIMA or any one of Ledgr's AI Models, input information and then adjust parameters of the engine suiting your requirement, prior to running the algorithms.
    - Please note that on your instruction, the AI model will execute in real-time.

    *Hence, sometimes it may seem to be taking long or the browser may have stalled. However, in reality the AI is running in the back-end.*
    - On completion, predicted prices are presented along with a set of other information.

    """)
st.write("  ------  ")

# st.header("DiscussionBox")

# - Discuss about your Portfolio, or your Wealth Journey at the DiscussionBox forums.

#- Share your Portfolios and or holdings, share opinions, observations, knowledge and most importantly, memes and degeneracy.
#- Content is organized by topics and threads, Users interact via Likes/Dislikes, Comments, shares and posts.
#- Users can build profiles, have friends and interact via the global "DiscussionWall" or across threads, groups, etc.
#- Users can earn LedgrTokens in a few ways other than purchasing them from the LedgrExchange portal.

#They are -

# (a) as initial rewards, intermittent grants and random AirDrops

# (b) by interacting on the platform and performing specific tasks [KYC, Referrals, Subscriptions to sub-plans etc.]

# (c) by participating in collab-work protocols, due-diligence-dropoffs for instance.

# - Threads are moderated, and some moderators shall be selected from active Users, to work along with experts.
# - Get great curated info, news, links, updates etc along with a variety of other content and resources.
# - Build the community, get engaged, grow together.
# """
t9, t10, t11 = st.columns([1, 5, 1])
with t9: st.write(" ")
with t10: st.caption(": | 2023 - 2024 | All Rights Resrved  ©  Ledgr Inc. | www.alphaLedgr.com | alphaLedgr Technologies Ltd. :")
with t11: st.write(" ")
