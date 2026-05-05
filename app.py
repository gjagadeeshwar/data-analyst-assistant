import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Data Analyst Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown('<div class="main-title">📈 Data Analyst Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload your data and ask questions in natural language.</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png", width=100) # Placeholder icon
    st.header("⚙️ Configuration")
    llm_choice = st.selectbox("Select LLM", ["Google Gemini", "OpenAI GPT-4o-mini"])
    
    st.markdown("---")
    st.write("**API Key Setup**")
    st.info("The app will automatically use keys from the `.env` file if available.")
    api_key = st.text_input("Override API Key (Optional)", type="password", help="Enter key here to override .env")
    
    st.markdown("---")
    st.write("Built with Streamlit & LangChain")

# --- Main Content ---
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the file
    with st.spinner("Loading data..."):
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Successfully loaded `{uploaded_file.name}`! ({df.shape[0]} rows, {df.shape[1]} columns)")
            
            # Data Preview Expander
            with st.expander("🔍 Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            st.divider()
            
            # Setup LLM
            llm = None
            key_to_use = api_key if api_key else (os.getenv("GOOGLE_API_KEY") if llm_choice == "Google Gemini" else os.getenv("OPENAI_API_KEY"))
            
            if not key_to_use or key_to_use.startswith("your_"):
                st.warning("⚠️ Please provide a valid API Key in the sidebar or update the `.env` file.")
            else:
                st.subheader("💬 Chat with your Data")
                st.write("Ask any question about the data, or request a summary/calculation.")
                
                if llm_choice == "Google Gemini":
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key_to_use, temperature=0)
                else:
                    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=key_to_use, temperature=0)
                
                # Create Agent
                agent = create_pandas_dataframe_agent(
                    llm, 
                    df, 
                    verbose=True, 
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )
                
                question = st.text_input("Enter your question:", placeholder="e.g., What is the average of the 'Sales' column?")
                
                if st.button("Analyze Data"):
                    if question:
                        with st.spinner("🧠 Analyzing your data..."):
                            try:
                                response = agent.invoke({"input": question})
                                st.markdown("### 📊 Result")
                                st.info(response["output"])
                            except Exception as e:
                                st.error(f"An error occurred during analysis: {e}")
                    else:
                        st.warning("Please enter a question first.")
                
        except Exception as e:
            st.error(f"Error processing the file: {e}")
else:
    st.info("👆 Please upload a file to get started.")
