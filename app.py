import pandas as pd
import streamlit as st



st.set_page_config(
    page_title="Sentiment analysis API",
    #layout='wide',
    initial_sidebar_state='expanded'
)

# ----------------- #
#      SIDEBAR      #
# ----------------- #



# -------------- #
#      BODY      #
# -------------- #

user_input = st.text_input('Input')