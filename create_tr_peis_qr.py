from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_H
from qrcode.image.svg import SvgImage


URL = "https://github.com/jaroslav-herman/TR-PEIS_procedure"
OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "tr_peis_procedure_qr.png"
SVG_PATH = OUTPUT_DIR / "tr_peis_procedure_qr.svg"
EISYFIT_URL = "https://pypi.org/project/eisyfit/"
EISYFIT_PNG_PATH = OUTPUT_DIR / "eisyfit_qr.png"
EISYFIT_SVG_PATH = OUTPUT_DIR / "eisyfit_qr.svg"


def build_qr_for_url(url: str) -> qrcode.QRCode:
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=16,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    return qr


def save_qr(url: str, png_path: Path, svg_path: Path) -> None:
    qr = build_qr_for_url(url)

    png = qr.make_image(fill_color="black", back_color="white")
    png.save(png_path)

    svg = qr.make_image(image_factory=SvgImage)
    svg.save(svg_path)


def main() -> None:
    save_qr(URL, PNG_PATH, SVG_PATH)
    save_qr(EISYFIT_URL, EISYFIT_PNG_PATH, EISYFIT_SVG_PATH)

    print(f"Created: {PNG_PATH}")
    print(f"Created: {SVG_PATH}")
    print(f"Encoded URL: {URL}")
    print(f"Created: {EISYFIT_PNG_PATH}")
    print(f"Created: {EISYFIT_SVG_PATH}")
    print(f"Encoded URL: {EISYFIT_URL}")


if __name__ == "__main__":
    main()
