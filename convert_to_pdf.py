#!/usr/bin/env python3
"""
Convert HTML guides to PDF using browser print functionality.
Run this script from the docs folder.

Requirements: pip install weasyprint
Alternative: Open HTML in browser and print to PDF (Ctrl+P)
"""

import os
import sys

def try_weasyprint():
    """Try converting with WeasyPrint."""
    try:
        from weasyprint import HTML, CSS

        css = CSS(string='''
            @page { size: A4; margin: 2cm; }
            body { font-family: sans-serif; font-size: 11pt; }
            h1 { page-break-before: always; }
            h1:first-of-type { page-break-before: avoid; }
            pre { white-space: pre-wrap; font-size: 9pt; }
            table { font-size: 10pt; }
        ''')

        files = [
            ('Radar_Simulation_Intermediate_Guide.html',
             'Radar_Simulation_Intermediate_Guide.pdf'),
            ('Radar_Simulation_Professional_Guide.html',
             'Radar_Simulation_Professional_Guide.pdf')
        ]

        for html_file, pdf_file in files:
            if os.path.exists(html_file):
                print(f"Converting {html_file} to PDF...")
                HTML(html_file).write_pdf(pdf_file, stylesheets=[css])
                print(f"  Created: {pdf_file}")
            else:
                print(f"  Warning: {html_file} not found")

        return True
    except ImportError:
        return False

def try_pdfkit():
    """Try converting with pdfkit (requires wkhtmltopdf)."""
    try:
        import pdfkit

        options = {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'encoding': 'UTF-8',
            'enable-local-file-access': None
        }

        files = [
            ('Radar_Simulation_Intermediate_Guide.html',
             'Radar_Simulation_Intermediate_Guide.pdf'),
            ('Radar_Simulation_Professional_Guide.html',
             'Radar_Simulation_Professional_Guide.pdf')
        ]

        for html_file, pdf_file in files:
            if os.path.exists(html_file):
                print(f"Converting {html_file} to PDF...")
                pdfkit.from_file(html_file, pdf_file, options=options)
                print(f"  Created: {pdf_file}")

        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"pdfkit error: {e}")
        return False

def main():
    print("=" * 60)
    print("Radar Simulation Guide - HTML to PDF Converter")
    print("=" * 60)

    # Try WeasyPrint first
    print("\nTrying WeasyPrint...")
    if try_weasyprint():
        print("\nSuccess! PDFs created with WeasyPrint.")
        return

    # Try pdfkit
    print("WeasyPrint not available. Trying pdfkit...")
    if try_pdfkit():
        print("\nSuccess! PDFs created with pdfkit.")
        return

    # Manual instructions
    print("\n" + "=" * 60)
    print("MANUAL CONVERSION REQUIRED")
    print("=" * 60)
    print("""
Neither WeasyPrint nor pdfkit is installed.

Option 1: Install WeasyPrint
    pip install weasyprint
    python convert_to_pdf.py

Option 2: Install pdfkit + wkhtmltopdf
    pip install pdfkit
    Download wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html
    python convert_to_pdf.py

Option 3: Manual conversion (recommended for best results)
    1. Open the HTML file in Chrome/Edge
    2. Press Ctrl+P (or Cmd+P on Mac)
    3. Select "Save as PDF" as the destination
    4. Click "More settings" and set:
       - Paper size: A4 or Letter
       - Margins: Default or Minimum
       - Enable "Background graphics"
    5. Click Save

Files to convert:
    - Radar_Simulation_Intermediate_Guide.html
    - Radar_Simulation_Professional_Guide.html
""")

if __name__ == '__main__':
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
