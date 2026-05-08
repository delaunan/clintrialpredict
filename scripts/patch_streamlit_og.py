import os
import streamlit as st
import sys

def patch_streamlit_index():
    # Find Streamlit's index.html
    streamlit_path = os.path.dirname(st.__file__)
    index_path = os.path.join(streamlit_path, "static", "index.html")
    
    if not os.path.exists(index_path):
        print(f"ERROR: Streamlit index.html not found at {index_path}")
        sys.exit(1)
        
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    og_block = """<!-- CTP_LINKEDIN_OPEN_GRAPH_START -->
<meta property="og:title" content="ClinTrialPredict | Clinical Trial Completion Risk" />
<meta property="og:description" content="Predicting clinical trial completion risk from 30,000+ Phase II/III trials using machine learning." />
<meta property="og:url" content="https://clintrial-ui-835962039082.europe-west1.run.app/" />
<meta property="og:type" content="website" />
<meta property="og:site_name" content="ClinTrialPredict" />
<meta property="og:image" content="https://clintrial-ui-835962039082.europe-west1.run.app/app/static/linkedin-preview-v2.png" />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="627" />
<!-- CTP_LINKEDIN_OPEN_GRAPH_END -->"""

    if "CTP_LINKEDIN_OPEN_GRAPH_START" in html_content:
        print("LinkedIn Open Graph tags already present. Skipping patch.")
        return

    if "</head>" not in html_content:
        print("ERROR: Could not find </head> tag in index.html")
        sys.exit(1)

    patched_content = html_content.replace("</head>", f"{og_block}\n</head>")
    
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(patched_content)
    
    print(f"Successfully patched {index_path}")

if __name__ == "__main__":
    patch_streamlit_index()
