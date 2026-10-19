#!/usr/bin/env python3
"""
FTPS client for octavo.troja.mff.cuni.cz
Downloads .mpt files from your server.
"""

import os
import argparse
from ftplib import FTP_TLS

def download_from_octavo(password, remote_file, local_path='.'):
    """
    Download from octavo.troja.mff.cuni.cz using FTPS.
    """
    HOST = 'octavo.troja.mff.cuni.cz'
    USERNAME = 'HermanJ'

    try:
        ftp = FTP_TLS()
        ftp.connect(HOST, 21, timeout=10)
        ftp.login(USERNAME, password)
        ftp.prot_p()  # REQUIRED for this server

        filename = os.path.basename(remote_file)
        local_file = os.path.join(local_path, filename)

        print(f"Downloading {remote_file}...")

        with open(local_file, 'wb') as f:
            ftp.retrbinary(f'RETR {remote_file}', f.write)

        ftp.quit()
        print(f"✓ Downloaded to: {local_file}")
        print(f"  Size: {os.path.getsize(local_file)} bytes")
        return local_file

    except Exception as e:
        print(f"✗ Error: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download from octavo.troja.mff.cuni.cz')
    parser.add_argument('--password', default = 'agamaci3', help='FTP password')
    parser.add_argument('--file', required=True, help='Remote file path')
    parser.add_argument('--local', default='.', help='Local directory')

    args = parser.parse_args()

    download_from_octavo(args.password, args.file, args.local)