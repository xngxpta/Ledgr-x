import pandas as pd
import datetime as dt
import streamlit as st
import os

direc = os.getcwd()
st.header("Contact, Suggestions & Communication")
logofile = f"{direc}/pages/appdata/imgs/2.png"

with st.sidebar:
    st.image(logofile, width=250)
st.subheader("Hi. Thanks for your time with Ledgr. Ledgr develops on active guidance from its Users and Visitors. Any suggestion and feedback is welcome!")
st.write("Please let us know about your experience and suggestions, starting with the Questions below:")
with st.form('Feedback'):
	name = st.text_input("Your Name")
	email = st.text_input("Your Email")
	phone = st.number_input("Your Contact Number for us to get back to you if need be, once we review your message")
	feedbck_1 = st.text_area("Please write in your message here!!")
	feedbck_2 = st.text_area("Any additional features or suggestions your may have for our teams?")
	exp_level = st.slider(label = 'Rate your Experience out of 10, with 10 being Awesome!', 
		min_value=1, max_value=10, value=6, step=1, help=None)
	submitted = st.form_submit_button("Submit")
	if submitted:
		df_feedback_2 = pd.DataFrame({"Name" : [name], "Email" : [email], "phone" : [phone], "feedbck_1" : [feedbck_1], "feedbck_2" : [feedbck_2], "exp_level" : [exp_level]})
		st.write(df_feedback_2)
		df_feedback_2.to_csv(f"{direc}/pages/appdata/User_Correspondences_tstamp.csv")



st.write("  -------------  ")
t9, t10, t11 = st.columns([1, 5, 1])
with t9: st.write(" ")
with t10: st.caption(": | 2023 - 2024 | All Rights Resrved  ©  Ledgr Inc. | www.alphaLedgr.com | alphaLedgr Technologies Ltd. :")
with t11: st.write(" ")
