import os
import time
import json
from typing import Optional
import boto3
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import io
from PyPDF2 import PdfReader
import tempfile

# Load environment variables
load_dotenv()

# Define environment variables or default values
S3_BUCKET = os.environ.get("S3_BUCKET")
if not S3_BUCKET:
    st.error("S3_BUCKET environment variable is not set!")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# Uncomment the model you wish to use:
# MODEL_ID = "amazon.nova-micro-v1:0"
MODEL_ID = "meta.llama3-3-70b-instruct-v1:0"

def upload_to_s3(file_obj, bucket, key):
    """Upload a file to S3"""
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3_client.upload_fileobj(file_obj, bucket, key)
        return True
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return False

def invoke_bedrock_model(client: boto3.client, prompt: str, temperature: float, top_p: float, max_gen_len: int) -> Optional[str]:
    """
    Invoke the Meta Llama 3 model using a properly formatted request payload.
    The expected payload includes only 'prompt' along with optional keys 'temperature',
    'top_p', and 'max_gen_len'.
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
        
        # Assuming the response contains a key "generation" with the generated text.
        if "generation" in response_body:
            return response_body["generation"]
        return ""
    
    except Exception as e:
        st.error(f"Error invoking model: {str(e)}")
        return ""

def process_document(s3_key, custom_prompt, temperature, top_p, max_gen_len):
    """
    Process a document by extracting text via Textract and then invoking Bedrock.
    
    Args:
        s3_key (str): S3 object key of the uploaded document.
        custom_prompt (str): Custom prompt for Bedrock analysis.
        temperature (float): Temperature parameter for text generation.
        top_p (float): Top P (nucleus sampling) parameter.
        max_gen_len (int): Maximum number of tokens to generate.
    
    Returns:
        dict: Contains extracted text, AI analysis result, and processing times.
    """
    try:
        # Initialize AWS clients
        textract_client = boto3.client("textract", region_name=AWS_REGION)
        bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

        document = {
            "S3Object": {
                "Bucket": S3_BUCKET,
                "Name": s3_key,
            }
        }

        # Process with Textract and measure time
        textract_start = time.time()
        with st.spinner('Processing document with Textract...'):
            detect_text_output = textract_client.detect_document_text(Document=document)
            extracted_text = "\n".join(
                [block["Text"] for block in detect_text_output["Blocks"] if "Text" in block]
            )
        textract_time = time.time() - textract_start

        # Combine custom prompt with the extracted text
        full_prompt = f"{custom_prompt}\n\nExtracted Text:\n{extracted_text}"

        # Process with Bedrock and measure time
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
        return {
            "extracted_text": "",
            "analysis_result": "",
            "textract_time": 0,
            "bedrock_time": 0
        }

def main():
    st.set_page_config(page_title="Document Analysis with AWS", layout="wide")
    
    st.title("Low Latency Document Analysis with AWS")
    st.write("Upload a document and analyze it using AWS Textract and Bedrock (Meta Llama 3)")

    # Layout: two columns for file input and process results
    col1, col2 = st.columns([1, 1])

    # Sidebar: Inference parameters
    with st.sidebar:
        st.header("Inference Parameters")
        max_gen_len = st.slider(
            label="Maximum tokens to generate",
            min_value=50,
            max_value=2000,
            value=512,
            step=50
        )
        temperature = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        top_p = st.slider(
            label="Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1
        )
        
    with col1:
        uploaded_file = st.file_uploader(
            label="Upload your document",
            type=['png', 'jpg', 'jpeg', 'pdf']
        )
        
        # Custom prompt input
        default_prompt = "Extract key information from the document and summarize it."
        custom_prompt = st.text_area(
            label="Enter your analysis prompt",
            value=default_prompt,
            height=150,
            help="Specify how you want the document to be analyzed"
        )

        # File preview handling for PDF and images
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
                        st.error("Multi-page documents are not supported for this demonstration. Please upload a single-page document.")
                        uploaded_file = None
                    else:
                        st.write("PDF document preview:")
                        page = pdf_reader.pages[0]
                        st.text_area(
                            label="PDF content",
                            value=page.extract_text(),
                            height=300,
                            disabled=True
                        )
                    
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    uploaded_file = None
                    
            elif file_type.startswith('image/'):
                st.write("Image preview:")
                st.image(uploaded_file, caption="Preview of uploaded document", use_container_width=True)

    with col2:
        if uploaded_file and st.button("Process Document", type="primary"):
            total_start = time.time()
            
            file_extension = uploaded_file.name.split('.')[-1]
            s3_key = f"uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"

            with st.spinner('Uploading file to S3...'):
                if upload_to_s3(uploaded_file, S3_BUCKET, s3_key):
                    st.success("File uploaded successfully!")
                    
                    result = process_document(s3_key, custom_prompt, temperature, top_p, max_gen_len)
                    total_time = time.time() - total_start
                    
                    # Display processing times
                    col1_metric, col2_metric, col3_metric = st.columns(3)
                    with col1_metric:
                        st.metric(label="Textract Processing Time", value=f"{result['textract_time']:.2f}s")
                    with col2_metric:
                        st.metric(label="Bedrock Analysis Time", value=f"{result['bedrock_time']:.2f}s")
                    with col3_metric:
                        st.metric(label="Total Processing Time", value=f"{total_time:.2f}s")
                    
                    # Display results
                    st.subheader("Extracted Text")
                    st.text_area(
                        label="Text extracted from document",
                        value=result['extracted_text'],
                        height=200,
                        key="extracted_text"
                    )
                    
                    st.subheader("Analysis Result")
                    st.text_area(
                        label="AI analysis results",
                        value=result['analysis_result'],
                        height=200,
                        key="analysis_result"
                    )
                else:
                    st.error("Failed to upload file")

if __name__ == "__main__":
    main()
