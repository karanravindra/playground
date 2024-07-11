from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string


def get_random_text(length=6):
    letters = string.ascii_uppercase + string.digits
    return "".join(random.choice(letters) for _ in range(length))


def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def create_captcha_image(text):
    width, height = 200, 100
    image = Image.new("RGB", (width, height), get_random_color())
    font = ImageFont.truetype("arial.ttf", 40)
    draw = ImageDraw.Draw(image)

    for i in range(random.randint(1000, 3000)):
        draw.point(
            [random.randint(0, width), random.randint(0, height)],
            fill=get_random_color(),
        )

    text_width, text_height = draw.textsize(text, font)
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, font=font, fill=get_random_color())

    image = image.transform(
        (width, height), Image.AFFINE, (1, 0.3, 0, 0.1, 1, 0), Image.BICUBIC
    )
    image = image.filter(ImageFilter.GaussianBlur(1))

    return image


if __name__ == "__main__":
    text = get_random_text()
    print(f"CAPTCHA Text: {text}")
    captcha_image = create_captcha_image(text)
    captcha_image.save("captcha.png")
