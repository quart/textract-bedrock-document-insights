import os
import time
import json
import logging
import sys
from datetime import datetime
import streamlit as st
import boto3
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import tempfile

# ---------------- Logger Configuration ----------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ---------------- Environment Setup ----------------
load_dotenv()
S3_BUCKET = os.environ.get("S3_BUCKET")
if not S3_BUCKET:
    st.error("S3_BUCKET environment variable is not set!")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = "meta.llama3-3-70b-instruct-v1:0"

# ---------------- Existing Functions ----------------
def upload_to_s3(file_obj, bucket, key):
    """Upload a file to S3"""
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3_client.upload_fileobj(file_obj, bucket, key)
        return True
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return False

def invoke_bedrock_model(client: boto3.client, prompt: str, temperature: float, top_p: float, max_gen_len: int) -> str:
    """
    Invoke the Meta Llama 3 model using a properly formatted request payload.
    The payload includes only 'prompt', 'temperature', 'top_p', and 'max_gen_len'.
    """
    request_body = {
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_gen_len": max_gen_len
    }
    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        response_body = json.loads(response['body'].read())
        logger.info("Raw Bedrock response: %s", response_body)
        return response_body.get("generation", "")
    except Exception as e:
        st.error(f"Error invoking model: {str(e)}")
        return ""

def process_document(s3_key, custom_prompt, temperature, top_p, max_gen_len):
    """
    Process a document by extracting text via Textract and then invoking Bedrock.
    """
    try:
        textract_client = boto3.client("textract", region_name=AWS_REGION)
        bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        document = {"S3Object": {"Bucket": S3_BUCKET, "Name": s3_key}}
        
        textract_start = time.time()
        with st.spinner('Processing document with Textract...'):
            detect_text_output = textract_client.detect_document_text(Document=document)
            extracted_text = "\n".join(
                [block["Text"] for block in detect_text_output["Blocks"] if "Text" in block]
            )
        textract_time = time.time() - textract_start
        
        full_prompt = f"{custom_prompt}\n\nExtracted Text:\n{extracted_text}"
        
        bedrock_start = time.time()
        with st.spinner('Analyzing with Bedrock...'):
            analysis_result = invoke_bedrock_model(bedrock_client, full_prompt, temperature, top_p, max_gen_len)
        bedrock_time = time.time() - bedrock_start
            
        return {
            "extracted_text": extracted_text,
            "analysis_result": analysis_result,
            "textract_time": textract_time,
            "bedrock_time": bedrock_time
        }
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return {"extracted_text": "", "analysis_result": "", "textract_time": 0, "bedrock_time": 0}

# ---------------- New Function to Save Analysis Results ----------------
def save_analysis_result(analysis_text: str, bucket: str, key: str) -> bool:
    """
    Save the AI analysis results as a text file to S3.
    """
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=analysis_text)
        return True
    except Exception as e:
        st.error(f"Error saving analysis results: {str(e)}")
        return False

# ---------------- Main Function ----------------
def main():
    st.set_page_config(page_title="Document Analysis with AWS", layout="wide")
    st.title("Low Latency Document Analysis with AWS")
    st.write("Upload a document and analyze it using AWS Textract and Bedrock (Meta Llama 3)")

    # Store results in session state to persist across reruns
    if 'result' not in st.session_state:
        st.session_state.result = None

    # Define two columns: col1 for uploader/prompt, col2 for results and buttons
    col1, col2 = st.columns([1, 1])
    
    with st.sidebar:
        st.header("Inference Parameters")
        max_gen_len = st.slider("Maximum tokens to generate", 50, 2000, 512, 50)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1)
        
    with col1:
        uploaded_file = st.file_uploader("Upload your document", type=['png', 'jpg', 'jpeg', 'pdf'])
        default_prompt = "Extract key information from the document and summarize it."
        custom_prompt = st.text_area("Enter your analysis prompt", value=default_prompt, height=150, 
                                     help="Specify how you want the document to be analyzed")
        if uploaded_file is not None:
            file_type = uploaded_file.type
            if file_type == "application/pdf":
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file.flush()
                        tmp_file_path = tmp_file.name
                    pdf_reader = PdfReader(tmp_file_path)
                    num_pages = len(pdf_reader.pages)
                    if num_pages > 1:
                        st.error("Multi-page documents are not supported. Please upload a single-page document.")
                        uploaded_file = None
                    else:
                        st.write("PDF document preview:")
                        page = pdf_reader.pages[0]
                        st.text_area("PDF content", value=page.extract_text(), height=300, disabled=True)
                    os.unlink(tmp_file_path)
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    uploaded_file = None
            elif file_type.startswith('image/'):
                st.write("Image preview:")
                st.image(uploaded_file, caption="Preview of uploaded document", use_container_width=True)
    
    with col2:
        # Place Process Document and Save Analysis buttons in one row
        btn_cols = st.columns(2)
        process_clicked = btn_cols[0].button("Process Document", type="primary")
        save_clicked = btn_cols[1].button("Save Analysis to S3")
        
        if process_clicked and uploaded_file:
            total_start = time.time()
            file_extension = uploaded_file.name.split('.')[-1]
            s3_key = f"uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
            with st.spinner('Uploading file to S3...'):
                if upload_to_s3(uploaded_file, S3_BUCKET, s3_key):
                    st.success("File uploaded successfully!")
                else:
                    st.error("Failed to upload file")
                    return
            result = process_document(s3_key, custom_prompt, temperature, top_p, max_gen_len)
            st.session_state.result = result
            total_time = time.time() - total_start
            col1_metric, col2_metric, col3_metric = st.columns(3)
            with col1_metric:
                st.metric("Textract Processing Time", f"{result['textract_time']:.2f}s")
            with col2_metric:
                st.metric("Bedrock Analysis Time", f"{result['bedrock_time']:.2f}s")
            with col3_metric:
                st.metric("Total Processing Time", f"{total_time:.2f}s")
        
        if st.session_state.result:
            st.subheader("Extracted Text")
            st.text_area("Text extracted from document", value=st.session_state.result['extracted_text'], height=200, key="extracted_text")
            st.subheader("Analysis Result")
            st.text_area("AI analysis results", value=st.session_state.result['analysis_result'], height=200, key="analysis_result")
        
        if save_clicked and st.session_state.result:
            analysis_key = f"analysis_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}_analysis.txt"
            logger.info("Save Analysis button pressed.")
            logger.info("Analysis result to be saved: %s", st.session_state.result['analysis_result'])
            if save_analysis_result(st.session_state.result['analysis_result'], S3_BUCKET, analysis_key):
                st.success(f"Analysis results saved successfully to {analysis_key}")
                logger.info("Analysis result saved to S3 with key: %s", analysis_key)
            else:
                st.error("Failed to save analysis results to S3.")
                logger.error("Failed to save analysis results to S3.")

if __name__ == "__main__":
    main()
