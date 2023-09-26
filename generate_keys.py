import streamlit as st
import pickle
from pathlib import Path
import yaml

import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
names = ["Riju Sengupta", "Sibani Sengupta", "Gautam Sengupta", "Prithviraj Sengupta", "User 7"]
usernames = ["r_xn", "sibani_s", "gsengupta56", "psengupta033", "user007"]

passwords == ['rx1988', 'ss1958', 'gs1956', 'ps1995', 'u0071']

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)