import os
import re
import sys

import streamlit as st


START_MARKER = "<!-- CTP_LINKEDIN_OPEN_GRAPH_START -->"
END_MARKER = "<!-- CTP_LINKEDIN_OPEN_GRAPH_END -->"

PUBLIC_APP_URL = os.getenv(
    "PUBLIC_APP_URL",
    "https://clintrial-ui-835962039082.europe-west1.run.app",
).rstrip("/")

OG_TITLE = "CTPredict | Predict Late-Stage Trial Completion & Early Termination | Key Drivers & Risk Tiers"

OG_DESCRIPTION = (
    "Predict full clinical trial completion from early trial design information. "
    "Built with machine learning trained on 30,000+ Phase II/III trials from "
    "publicly available clinical trial data. Explore risk tiers, score drivers, "
    "and benchmarked operational signals."
)
OG_IMAGE_FILENAME = "linkedin-preview-v5.png"
OG_IMAGE_URL = f"{PUBLIC_APP_URL}/app/static/{OG_IMAGE_FILENAME}"


def patch_streamlit_index():
    streamlit_path = os.path.dirname(st.__file__)
    index_path = os.path.join(streamlit_path, "static", "index.html")

    if not os.path.exists(index_path):
        print(f"ERROR: Streamlit index.html not found at {index_path}")
        sys.exit(1)

    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    og_block = f"""{START_MARKER}
<meta name="description" content="{OG_DESCRIPTION}" />
<meta property="og:title" content="{OG_TITLE}" />
<meta property="og:description" content="{OG_DESCRIPTION}" />
<meta property="og:url" content="{PUBLIC_APP_URL}/" />
<meta property="og:type" content="website" />
<meta property="og:site_name" content="ClinTrialPredict" />
<meta property="og:image" content="{OG_IMAGE_URL}" />
<meta property="og:image:secure_url" content="{OG_IMAGE_URL}" />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="627" />
<meta property="og:image:type" content="image/png" />

{END_MARKER}"""

    if START_MARKER in html_content and END_MARKER in html_content:
        existing_og_pattern = (
            rf"\n?{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}\n?"
        )
        html_content = re.sub(
            existing_og_pattern,
            "\n",
            html_content,
            flags=re.DOTALL,
        )

    if "</head>" not in html_content:
        print("ERROR: Could not find </head> tag in index.html")
        sys.exit(1)

    patched_content = html_content.replace("</head>", f"{og_block}\n</head>", 1)

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(patched_content)

    print(f"Successfully patched {index_path}")
    print(f"Open Graph image URL: {OG_IMAGE_URL}")


if __name__ == "__main__":
    patch_streamlit_index()
