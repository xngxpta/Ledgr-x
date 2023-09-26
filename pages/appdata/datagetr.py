#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:50:15 2022

@author: r_xngxpta
"""
import pandas as pd
import datetime as dt
import yfinance as yf
import yahoo_fin as yfin
from nsepy import get_history as gh
import pandas_datareader as web
import os
import streamlit as st
direc = os.getcwd()
st.write(direc)
tickerfile = pd.read_csv(f'{direc}/EQUITY_L.csv')
st.write(tickerfile.tail())
tickerlist = tickerfile['SYMBOL']
for x in tickerlist:
	yticker = x + '.NS'
	path = f'{direc}/OHLC/{yticker}_data.csv'
	if not os.path.exists(path):
		# st.write("{x} datafile exists")
		data = yf.download(yticker)
		data.to_csv(path)
		(f"Data for stock {x} gathered and csv file generated")
	else:
		st.write(f'{x} Datafile exists!!')


# pfhome = f'{direc}/userinfo/userportfolios/{username}_Ledgr.csv'