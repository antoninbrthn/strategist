import PIL
from PIL import ImageDraw


def build_info_panel(width, height, text_content, font=None):
    """
    Create a black rectangle of (width x height),
    and draw multiline text in white.
    """
    panel = PIL.Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.multiline_text((10, 10), text_content, fill=(255, 255, 255), font=font)
    return panel
