#!/usr/bin/env python3
from ftplib import FTP_TLS

# Your credentials
HOST = 'octavo.troja.mff.cuni.cz'
USERNAME = 'HermanJ'
PASSWORD = 'agamaci3'  # Replace with your password
REMOTE_FILE = 'PEM-WE/PEM_WE_research_new_PC/2026/482_II_IrRuOx_N115_RDC1152_none/II_Day10_procedure1_03_SV_C02.mpt'  # Replace with your file path

# Connect using FTPS
ftp = FTP_TLS()
ftp.connect(HOST, 21, timeout=10)
ftp.login(USERNAME, PASSWORD)
ftp.prot_p()  # REQUIRED: Switch to secure data channel

# Download file
with open('downloaded_file.mpt', 'wb') as f:
    ftp.retrbinary(f'RETR {REMOTE_FILE}', f.write)

ftp.quit()
print("✓ File downloaded successfully!")