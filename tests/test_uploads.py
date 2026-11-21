import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from PIL import Image
except ImportError:   # pragma: no cover
    Image = None

try:
    import pillow_heif
except ImportError:   # pragma: no cover
    pillow_heif = None

import utils


def _jpeg(size, orientation=None) -> bytes:
    image = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        image.save(buf, "JPEG", exif=exif)
    else:
        image.save(buf, "JPEG")
    return buf.getvalue()


@unittest.skipIf(Image is None, "Pillow is not installed")
class NormalizeImageTests(unittest.TestCase):
    def test_sideways_phone_photo_is_rotated_upright(self):
        data = _jpeg((300, 200), orientation=6)   # EXIF: "rotate 90 degrees to view"
        out, ext = utils.normalize_image(data, ".jpg")
        self.assertEqual(ext, ".jpg")
        self.assertEqual(Image.open(io.BytesIO(out)).size, (200, 300))

    def test_huge_photo_is_downscaled(self):
        buf = io.BytesIO()
        Image.new("RGB", (5000, 3000), "white").save(buf, "PNG")
        out, ext = utils.normalize_image(buf.getvalue(), ".png")
        self.assertEqual(ext, ".jpg")
        self.assertEqual(max(Image.open(io.BytesIO(out)).size), utils.MAX_IMAGE_SIDE_PX)

    def test_fine_image_is_left_byte_for_byte_alone(self):
        buf = io.BytesIO()
        Image.new("RGB", (100, 80), "white").save(buf, "PNG")
        out, ext = utils.normalize_image(buf.getvalue(), ".png")
        self.assertEqual((out, ext), (buf.getvalue(), ".png"))

    @unittest.skipIf(pillow_heif is None, "pillow-heif is not installed")
    def test_iphone_heic_becomes_jpeg(self):
        pillow_heif.register_heif_opener()
        buf = io.BytesIO()
        Image.new("RGB", (120, 90), (10, 120, 200)).save(buf, "HEIF")
        out, ext = utils.normalize_image(buf.getvalue(), ".heic")
        self.assertEqual(ext, ".jpg")
        self.assertEqual(Image.open(io.BytesIO(out)).format, "JPEG")

    def test_transparent_png_that_needs_resizing_gets_a_white_background(self):
        buf = io.BytesIO()
        Image.new("RGBA", (4000, 100), (0, 0, 0, 0)).save(buf, "PNG")
        out, _ = utils.normalize_image(buf.getvalue(), ".png")
        self.assertEqual(Image.open(io.BytesIO(out)).getpixel((5, 5)), (255, 255, 255))

    def test_corrupt_image_is_a_clean_error(self):
        with self.assertRaises(utils.UnsupportedFileError):
            utils.normalize_image(b"definitely not an image", ".jpg")

    def test_heic_without_pillow_heif_says_what_to_install(self):
        with mock.patch.dict("sys.modules", {"pillow_heif": None}):
            with self.assertRaises(utils.UnsupportedFileError) as ctx:
                utils.normalize_image(b"x", ".heic")
        self.assertIn("pillow-heif", str(ctx.exception))


@unittest.skipIf(Image is None, "Pillow is not installed")
class SaveUploadedFileTests(unittest.TestCase):
    def test_heic_upload_is_stored_as_jpeg_and_non_images_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(utils, "UPLOAD_DIR", Path(tmp)):
            path = utils.save_uploaded_file(_jpeg((300, 200), orientation=6), "page.JPG")
            self.assertEqual(path.suffix, ".jpg")
            with Image.open(path) as saved:
                self.assertEqual(saved.size, (200, 300))

            pdf = utils.save_uploaded_file(b"%PDF-1.4 not really", "scan.pdf")
            self.assertEqual((pdf.suffix, pdf.read_bytes()), (".pdf", b"%PDF-1.4 not really"))

    def test_heic_is_an_allowed_upload_type(self):
        utils.validate_upload("IMG_0001.HEIC", 1000)
        self.assertEqual(utils.guess_mime_type(Path("a.heic")), "image/heic")


if __name__ == "__main__":
    unittest.main()
