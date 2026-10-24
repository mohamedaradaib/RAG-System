import asyncio
import hashlib
import time
from urllib.parse import quote

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

from vector_db import QdrantStorage

load_dotenv()

st.set_page_config(page_title="RAG System", page_icon="", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stHeaderActionElements"] {
        visibility: hidden !important;
    }
    :root {
        --gray-950: #191b1d;
        --gray-900: #202224;
        --gray-800: #292c2f;
        --gray-700: #2b2f36;
        --gray-600: #3a3f46;
        --gray-500: #6b7280;
        --gray-300: #d1d5db;
        --gray-100: #f3f4f6;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp, section.main, .main {
        background: var(--gray-700) !important;
        color: var(--gray-100) !important;
    }
    .block-container {
        padding-top: 5rem;
        padding-bottom: 4rem;
        background: var(--gray-700) !important;
    }
    .st-key-app-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        box-sizing: border-box;
        padding: 0.5rem 1.5rem;
        background: var(--gray-950);
        border-bottom: 1px solid rgba(209, 213, 219, 0.12);
        z-index: 1001;
    }
    .st-key-app-header [data-testid="stHorizontalBlock"] {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        align-items: center;
        gap: 0.5rem;
    }
    .st-key-app-header [data-testid="stColumn"] {
        width: 100% !important;
        min-width: 0 !important;
        flex: none !important;
    }
    .st-key-app-header [data-testid="stPopoverButton"],
    .st-key-app-header .stLinkButton,
    .st-key-app-header .stLinkButton > a {
        width: auto !important;
        box-sizing: border-box;
        justify-content: center;
        white-space: nowrap;
    }
    .st-key-app-header [data-testid="stPopover"] {
        width: max-content !important;
    }
    .st-key-app-header [data-testid="stColumn"]:last-child {
        display: flex;
        justify-content: flex-end;
    }
    .st-key-app-header [data-testid="stColumn"]:last-child .stLinkButton {
        display: flex;
        justify-content: flex-end;
    }
    .st-key-app-header .stLinkButton > a {
        background: transparent !important;
        border: 0 !important;
        padding: 0.6rem 1.1rem !important;
        box-shadow: none !important;
        transition: color 0.15s;
    }
    .st-key-app-header .stLinkButton > a:hover,
    .st-key-app-header .stLinkButton > a:focus-visible {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        text-decoration: none !important;
    }
    .st-key-app-header .header-nav-button {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        min-height: 2.5rem;
        box-sizing: border-box;
        padding: 0.6rem 0.5rem;
        border: 1px solid rgba(244, 246, 247, 0.32);
        border-radius: 0.5rem;
        color: #f4f6f7 !important;
        font-size: 0.875rem;
        text-decoration: none !important;
        white-space: nowrap;
    }
    .st-key-app-header .header-nav-button:hover,
    .st-key-app-header .header-nav-button:focus-visible {
        background: rgba(244, 246, 247, 0.1);
        border-color: rgba(244, 246, 247, 0.65);
    }
    .st-key-app-header .header-title {
        font-size: 1.5rem;
        line-height: 1.2;
        font-weight: 700;
        text-align: center;
        white-space: nowrap;
    }
    .st-key-app-header [data-testid="stMarkdownContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stPopover"] {
        position: relative !important;
        top: auto;
        right: auto;
        width: 100% !important;
        z-index: 1000000;
    }
    [data-testid="stPopoverButton"] {
        background: var(--gray-950) !important;
        color: var(--gray-100) !important;
        border: 1px solid rgba(209, 213, 219, 0.18) !important;
        border-radius: 10px !important;
    }
    [data-testid="stPopoverButton"]:hover {
        background: var(--gray-800) !important;
        border-color: rgba(209, 213, 219, 0.36) !important;
    }
    [data-testid="stSidebar"] {
        background: var(--gray-900) !important;
        color: var(--gray-100) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--gray-100) !important;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }
    [data-testid="stFileUploader"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    [data-testid="stFileUploader"] > label,
    .stTextInput > label,
    .stTextArea > label,
    .stSlider > label,
    .stSelectbox > label,
    .stNumberInput > label {
        color: var(--gray-100) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        width: min(100%, 15rem);
        aspect-ratio: 1;
        min-height: 0 !important;
        margin: 0 auto;
        padding: 1rem;
        gap: 0.75rem;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1.5px dashed rgba(209, 213, 219, 0.4);
        border-radius: 16px;
        background: var(--gray-900) !important;
        box-shadow: inset 0 0 0 1px rgba(180, 180, 180, 0.12);
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    .stButton > button,
    .stLinkButton > a,
    [data-testid="stFormSubmitButton"] button {
        background: var(--gray-950) !important;
        color: var(--gray-100) !important;
        border: 1px solid rgba(209, 213, 219, 0.18) !important;
        border-radius: 10px !important;
        font-weight: 600;
        padding: 0.6rem 1.1rem;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    .stButton > button:hover,
    .stLinkButton > a:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: var(--gray-800) !important;
        border-color: rgba(209, 213, 219, 0.36) !important;
    }
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox select,
    .stNumberInput input {
        background: var(--gray-900) !important;
        color: var(--gray-100) !important;
        border: 1px solid rgba(209, 213, 219, 0.2) !important;
        border-radius: 10px;
    }
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: rgba(209, 213, 219, 0.7) !important;
    }
    [data-testid="stSlider"] [data-rac][data-orientation="horizontal"] > div[data-rac] > div:first-child {
        background: var(--gray-100) !important;
    }
    [data-testid="stSlider"] [data-rac][data-orientation="horizontal"] > div[data-rac] > div[data-rac] {
        background: var(--gray-100) !important;
        border: 2px solid var(--gray-900) !important;
    }
    [data-testid="stSlider"] [data-rac][data-orientation="horizontal"] > div[data-rac] > div[data-rac]:active,
    [data-testid="stSlider"] [data-rac][data-orientation="horizontal"] > div[data-rac] > div[data-rac]:focus-within {
        background: var(--gray-600) !important;
        box-shadow: 0 0 0 3px rgba(58, 63, 70, 0.55) !important;
    }
    h1, h2 {
        text-align: center;
    }
    h1 {
        font-size: 1.5rem;
        line-height: 1.2;
        letter-spacing: 0;
        font-weight: 700;
        margin: 0 0 1rem 0;
        color: var(--gray-100);
    }
    html, body, .stApp, [data-testid="stAppViewContainer"], section.main, .main,
    [data-testid="stMain"] {
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    html::-webkit-scrollbar, body::-webkit-scrollbar, .stApp::-webkit-scrollbar,
    [data-testid="stAppViewContainer"]::-webkit-scrollbar, section.main::-webkit-scrollbar,
    .main::-webkit-scrollbar, [data-testid="stMain"]::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
    }
    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        box-sizing: border-box;
        padding: 0.75rem;
        text-align: center;
        font-size: 0.8rem;
        color: var(--gray-300);
        background: var(--gray-950);
        border-top: 1px solid rgba(209, 213, 219, 0.12);
        z-index: 1000;
    }
    :root {
        --canvas: #f4f6f7;
        --surface: #ffffff;
        --ink: #242b30;
        --muted: #647078;
        --line: #d9e0e4;
        --accent: #424b50;
        --accent-hover: #30383d;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp, section.main, .main {
        background: var(--canvas) !important;
        color: var(--ink) !important;
    }
    .block-container {
        max-width: 76rem;
        padding-top: 5.5rem;
        padding-bottom: 4rem;
        background: var(--canvas) !important;
    }
    .st-key-app-header {
        padding: 0.55rem 1.5rem;
        background: #242b30;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 3px 14px rgba(36, 43, 48, 0.12);
    }
    .st-key-app-header .header-title {
        color: #f4f6f7 !important;
        font-size: 1.2rem;
        font-weight: 650;
    }
    .st-key-app-header .stButton > button {
        background: transparent !important;
        border-color: rgba(244, 246, 247, 0.32) !important;
        color: #f4f6f7 !important;
    }
    .st-key-app-header .stButton > button *,
    .st-key-app-header .stLinkButton > a,
    .st-key-app-header .stLinkButton > a * {
        color: #f4f6f7 !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background: #e9edef !important;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        width: 100%;
        aspect-ratio: auto;
        min-height: 8rem !important;
        border: 1px dashed #aab5ba;
        border-radius: 12px;
        background: var(--surface) !important;
        box-shadow: none;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
        background: #f8fafb !important;
    }
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox select,
    .stNumberInput input {
        background: var(--surface) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px;
    }
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--muted) !important;
    }
    .stTextInput label,
    .stTextArea label,
    .stSlider label,
    [data-testid="stFileUploader"] > label {
        color: var(--ink) !important;
    }
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #68757c !important;
        box-shadow: 0 0 0 2px rgba(66, 75, 80, 0.14) !important;
    }
    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        transition: background 0.15s ease, transform 0.15s ease;
    }
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
        transform: translateY(-1px);
    }
    [data-testid="stPopover"] [data-testid="stPopoverButton"] {
        background: transparent !important;
        border: 1px solid rgba(243, 246, 242, 0.32) !important;
        color: #f3f6f2 !important;
    }
    [data-testid="stPopover"] [data-testid="stPopoverButton"] * {
        color: #f3f6f2 !important;
    }
    [data-testid="stPopoverBody"] {
        background: var(--surface) !important;
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    [data-testid="stForm"] {
        padding: 1rem 1rem 0.25rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    [data-testid="stSidebar"] h2 {
        text-align: left;
        font-size: 1rem;
    }
    .page-intro {
        margin: 1rem 0 1.25rem;
        padding: 1.25rem 0 1rem;
        border-bottom: 1px solid var(--line);
    }
    .page-intro .eyebrow {
        margin-bottom: 0.35rem;
        color: var(--accent) !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .page-intro h1 {
        margin: 0;
        color: var(--ink) !important;
        font-size: 1.8rem;
        text-align: left;
    }
    .fixed-footer {
        color: #d5dde1 !important;
        background: #242b30;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }
    [data-testid="stSlider"] [data-rac][data-orientation="horizontal"] > div[data-rac] > div:first-child {
        background: #d7d7d5 !important;
    }
    [data-testid="stSlider"] [data-rac][data-orientation="horizontal"] > div[data-rac] > div[data-rac] {
        background: var(--accent) !important;
        border: 2px solid #ffffff !important;
    }
    [data-testid="stSlider"] [data-rac][data-orientation="horizontal"] > div[data-rac] > div[data-rac]:active,
    [data-testid="stSlider"] [data-rac][data-orientation="horizontal"] > div[data-rac] > div[data-rac]:focus-within {
        background: var(--accent-hover) !important;
        box-shadow: 0 0 0 3px rgba(66, 66, 65, 0.2) !important;
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 5rem 1rem 4rem;
        }
        .st-key-app-header {
            padding: 0.45rem 0.65rem;
        }
        .st-key-app-header [data-testid="stHorizontalBlock"] {
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
            gap: 0.2rem;
        }
        .st-key-app-header .header-title {
            font-size: 0.95rem;
        }
        .st-key-app-header .stLinkButton > a {
            padding: 0.45rem 0.3rem !important;
            font-size: 0.8rem;
        }
        .page-intro h1 {
            font-size: 1.5rem;
        }
    }
    :root {
        --canvas: #e7e7e7;
        --surface: #ffffff;
        --ink: #191919;
        --muted: #4f4f4f;
        --line: #bdbdbd;
        --accent: #242424;
        --accent-hover: #090909;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp, section.main, .main {
        background: linear-gradient(145deg, #f0f0f0, var(--canvas)) !important;
        color: var(--ink) !important;
    }
    .block-container {
        max-width: 72rem;
        padding-top: 3.75rem;
        padding-bottom: 5rem;
        background: transparent !important;
    }
    .st-key-app-header {
        padding: 0.65rem 1.5rem;
        background: #1c1c1c;
        border-bottom: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
    }
    .st-key-app-header .header-title {
        color: #ffffff !important;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .st-key-app-header .stButton > button {
        border-color: rgba(255, 255, 255, 0.32) !important;
        border-radius: 7px !important;
    }
    [data-testid="stSidebar"] {
        background: #dedede !important;
        border-right: 1px solid var(--line);
    }
    [data-testid="stFileUploader"] {
        align-items: center;
    }
    [data-testid="stFileUploaderDropzone"] {
        width: auto;
        min-height: 0 !important;
        margin-inline: auto;
        padding: 0;
        border: 0;
        border-radius: 0;
        background: transparent !important;
        box-shadow: none;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: transparent;
        background: transparent !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        width: 15rem;
        height: 6rem;
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
    .upload-hint,
    .upload-formats {
        margin: 0.25rem 0;
        color: var(--muted);
        text-align: center;
    }
    .upload-title {
        margin: 3rem 0 0 !important;
        color: var(--ink) !important;
        font-size: clamp(1.5rem, 4vw, 2rem) !important;
        font-weight: 700;
        line-height: 1.2;
        text-align: center;
    }
    .upload-hint {
        margin-top: 0.75rem !important;
        margin-bottom: 1.25rem !important;
    }
    .upload-formats {
        margin-bottom: 2.5rem !important;
    }
    .st-key-suggest_summary button,
    .st-key-suggest_dates button,
    .st-key-suggest_compare button {
        min-height: 2.65rem;
        background: #ffffff !important;
        color: var(--accent) !important;
        border-color: #999999 !important;
        font-size: 0.84rem;
    }
    .st-key-suggest_summary button:hover,
    .st-key-suggest_dates button:hover,
    .st-key-suggest_compare button:hover {
        background: #e3e3e3 !important;
        border-color: var(--accent) !important;
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stAppViewContainer"] h3 {
        color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"] h3 {
        font-size: 1.08rem;
        font-weight: 700;
    }
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox select,
    .stNumberInput input {
        border-radius: 7px;
        border-color: #a9a9a9 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
    }
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.17) !important;
    }
    .stButton > button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stFileUploaderDropzone"] button {
        border-radius: 7px !important;
        font-weight: 650;
        transition: background 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
    }
    [data-testid="stFormSubmitButton"] button {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.22);
    }
    .stButton > button:hover,
    [data-testid="stFileUploaderDropzone"] button:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.16);
    }
    [data-testid="stForm"] {
        padding: 1.25rem 1.25rem 0.5rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    [data-testid="stAlert"] {
        background: #e2e2e2 !important;
        border: 1px solid var(--line) !important;
        color: var(--ink) !important;
    }
    [data-testid="stAlert"] * {
        color: var(--ink) !important;
    }
    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        box-sizing: border-box;
        padding: 0.75rem 1rem;
        color: #d5dde1 !important;
        background: #242b30;
        border-top: 1px solid rgba(209, 213, 219, 0.12);
        font-size: 0.75rem;
        z-index: 1000;
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 1.25rem 1rem 1.5rem;
        }
        .st-key-app-header {
            padding: 0.5rem 0.6rem;
        }
        .st-key-app-header .header-title {
            font-size: 0.95rem;
        }
        [data-testid="stForm"] {
            padding: 0.9rem 0.85rem 0.35rem;
        }
        .fixed-footer {
            padding: 0.5rem 0.25rem;
            line-height: 1.4;
        }
    }
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main {
        height: 100vh !important;
        overflow: hidden !important;
    }
    .block-container {
        height: 100vh;
        padding-top: 4.25rem !important;
        padding-bottom: 3.1rem !important;
        overflow: hidden !important;
    }
    .block-container > [data-testid="stVerticalBlock"] {
        height: 100%;
    }
    .block-container > [data-testid="stVerticalBlock"] > div:has(> .st-key-upload-area) {
        flex: 1 1 auto !important;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }
    .block-container > [data-testid="stVerticalBlock"] > div:has(> .st-key-chat-area) {
        flex: 0 0 auto !important;
    }
    .st-key-upload-area {
        flex: 1 1 auto !important;
        height: auto !important;
        justify-content: space-evenly;
        align-items: center;
        padding: 0;
    }
    .st-key-upload-area [data-testid="stElementContainer"],
    .st-key-upload-area [data-testid="stMarkdownContainer"] {
        height: auto !important;
    }
    .st-key-upload-area p, .st-key-upload-area h1 { line-height: 1.3 !important; }
    .st-key-upload-area > * {
        width: 100%;
    }
    .st-key-chat-area .stButton button:disabled:hover,
    .st-key-chat-area .stButton button:disabled:focus {
        background: transparent !important;
        border-color: rgba(49, 51, 63, 0.2) !important;
        transform: none !important;
        box-shadow: none !important;
        cursor: not-allowed;
    }
    .st-key-chat-area .stButton button { min-height: 2.25rem; padding: 0.25rem 0.75rem; }
    .upload-title { margin: 0 !important; font-size: clamp(1.1rem, 3.6vh, 2rem) !important; }
    .upload-hint { margin: 0 0 clamp(0.2rem, 1.2vh, 0.8rem) !important; font-size: clamp(0.8rem, 2vh, 1rem); }
    .upload-formats { margin: clamp(0.2rem, 1.2vh, 0.8rem) 0 0 !important; font-size: clamp(0.7rem, 1.8vh, 0.9rem); }
    [data-testid="stFileUploaderDropzone"] button {
        width: clamp(10rem, 36vh, 20rem);
        height: clamp(2.75rem, 14vh, 8rem);
        padding: 0.5rem 1rem;
    }
    [data-testid="stFileUploaderDropzone"] button + * { display: none; }
    [data-testid="stVerticalBlock"] { gap: clamp(0.15rem, 0.8vh, 0.5rem) !important; }
    .stApp h3 { margin: 0 !important; padding: 0 !important; font-size: clamp(0.95rem, 2.4vh, 1.25rem) !important; }
    [data-testid="stForm"] { padding: clamp(0.25rem, 1vh, 0.75rem) 1rem 0.25rem !important; }
    [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { display: none !important; }
    .st-key-chat-box {
        max-height: 100%;
        overflow-y: auto;
        scrollbar-width: thin;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    is_production = os.getenv("INNGEST_IS_PRODUCTION", "false").lower() == "true"
    return inngest.Inngest(app_id="rag_app", is_production=is_production)


@st.cache_resource
def get_qdrant_storage() -> QdrantStorage:
    return QdrantStorage()


def upload_document(file) -> str:
    api_base = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    headers = {"X-File-Name": quote(file.name, safe="")}
    api_token = os.getenv("APP_API_TOKEN")
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    response = requests.post(
        f"{api_base}/upload",
        data=file.getvalue(),
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["event_id"]


def render_citations(citations: list[dict]) -> None:
    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            location = (
                f"Page {citation['page']}"
                if citation.get("page") is not None
                else f"Chunk {citation['chunk_index'] + 1}"
                if citation.get("chunk_index") is not None
                else ""
            )
            st.text(f"[{citation['id']}] {citation['source']} {location}".strip())
            st.text(citation["text"])


header_slot = st.empty()
with header_slot.container(key="app-header"):
    indexed_column, title_column, contact_column = st.columns([2, 4, 1])
    with indexed_column:
        with st.popover("Files", use_container_width=True):
            try:
                qdrant_storage = get_qdrant_storage()
                indexed_sources = qdrant_storage.list_sources()
            except Exception:
                qdrant_storage = None
                indexed_sources = []
                index_unavailable = True
            else:
                index_unavailable = False

            if index_unavailable:
                st.caption("The document index is unavailable. Start Qdrant to manage indexed files.")
            elif indexed_sources:
                st.caption(f"{len(indexed_sources)} indexed document{'s' if len(indexed_sources) != 1 else ''}")
                for source in indexed_sources:
                    source_column, delete_column = st.columns([4, 1], vertical_alignment="center")
                    source_column.text(source)
                    if delete_column.button(
                        "Delete",
                        key=f"delete_source_{source}",
                        help=f"Remove {source} from the vector index",
                    ):
                        try:
                            qdrant_storage.delete_source(source)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Couldn't remove {source}: {exc}")
            else:
                st.caption("No files indexed")
    with title_column:
        st.markdown('<div class="header-title">RAG System</div>', unsafe_allow_html=True)
    with contact_column:
        st.link_button("Contact", "mailto:mohamedaradaib1@gmail.com", use_container_width=True)

with st.container(key="upload-area"):
    st.markdown('<h1 class="upload-title">Retrieval-Augmented Generation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="upload-hint">Drag and drop or upload files here.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Add documents",
        type=["pdf", "docx", "html", "htm", "txt", "md", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown(
        '<p class="upload-formats">Supported formats: PDF, DOCX, HTML, Markdown, TXT, CSV.</p>',
        unsafe_allow_html=True,
    )

async def send_rag_query_event(question: str, top_k: int, chat_history: list[dict] | None = None) -> None:
    client = get_inngest_client()
    payload = {
        "question": question,
        "top_k": top_k,
    }
    if chat_history:
        payload["chat_history"] = chat_history

    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data=payload,
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    is_production = os.getenv("INNGEST_IS_PRODUCTION", "false").lower() == "true"
    default = "https://api.inngest.com/v1" if is_production else "http://127.0.0.1:8288/v1"
    return os.getenv("INNGEST_API_BASE", default).rstrip("/")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    api_key = os.getenv("INNGEST_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                failure_reason = run.get("error") or run.get("output") or "Function run failed"
                raise RuntimeError(str(failure_reason))
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()

upload_completed = False
if not uploaded:
    st.session_state.processed_uploads.clear()
else:
    for file in uploaded:
        fingerprint = (file.name, hashlib.sha256(file.getvalue()).hexdigest())
        if fingerprint in st.session_state.processed_uploads:
            continue

        try:
            with st.spinner(f"Processing {file.name}..."):
                event_id = upload_document(file)
                output = wait_for_run_output(event_id)
            st.session_state.processed_uploads.add(fingerprint)
            if output.get("error"):
                st.session_state.setdefault("upload_notices", []).append(
                    ("error", f"Couldn't index {file.name}: {output['error']}")
                )
            else:
                st.session_state.setdefault("upload_notices", []).append(
                    ("success", f"Indexed {file.name}: {output.get('ingested', 0)} chunks.")
                )
                upload_completed = True
        except Exception as exc:
            st.session_state.processed_uploads.add(fingerprint)
            st.session_state.setdefault("upload_notices", []).append(
                ("error", f"Couldn't process {file.name}: {exc}")
            )

if upload_completed:
    st.rerun()

for notice_level, upload_notice in st.session_state.pop("upload_notices", []):
    getattr(st, notice_level)(upload_notice)

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_area = st.container(key="chat-area")
with chat_area:
    conversation_column, clear_column = st.columns([4, 2])
    with conversation_column:
        st.subheader("Conversation")
    with clear_column:
        if st.button(
            "Clear chat",
            disabled=not st.session_state.messages,
            help="Clear this conversation",
            use_container_width=True,
        ):
            st.session_state.messages.clear()
            st.rerun()

    chat_box = st.container(key="chat-box")
    with chat_box:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("citations"):
                    render_citations(message["citations"])
                elif message.get("sources"):
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.write(f"- {source}")

    with st.form("rag_query_form"):
        question = st.text_input(
            "Question",
            placeholder="Upload and index a document to start asking",
            key="rag_question_input",
            disabled=not indexed_sources,
        )
        settings_column, ask_column = st.columns([4, 1], vertical_alignment="bottom")
        with settings_column:
            top_k = st.slider("Retrieval depth", min_value=1, max_value=10, value=1, step=1)
        with ask_column:
            ask_submitted = st.form_submit_button(
                "Ask",
                use_container_width=True,
                disabled=not indexed_sources,
            )

if ask_submitted and question.strip():
    prompt = question.strip()
    query = prompt
else:
    prompt = None
    query = None

if query:
    st.session_state.messages.append({"role": "user", "content": prompt})
    conversation_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
        if msg["role"] in {"user", "assistant"}
    ]

    with chat_box, st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching..."):
        try:
            event_id = asyncio.run(send_rag_query_event(query, top_k, conversation_history))
            output = wait_for_run_output(event_id)
        except Exception as exc:
            output = {"answer": f"The AI service is currently unavailable: {exc}", "sources": [], "citations": []}

        answer = output.get("answer", "")
        sources = output.get("sources", [])
        citations = output.get("citations", [])

    assistant_message = {
        "role": "assistant",
        "content": answer or "I couldn't find relevant information in the uploaded documents.",
        "sources": sources,
        "citations": citations,
    }
    st.session_state.messages.append(assistant_message)
    st.rerun()

st.markdown('<div class="fixed-footer">© 2026 RAG System | About | Terms of Service | Privacy Policy | Security | Disclosure</div>', unsafe_allow_html=True)