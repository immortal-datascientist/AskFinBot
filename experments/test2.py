from PIL import Image
import pytesseract

image = Image.open("images/capture.PNG").convert("L")
text = pytesseract.image_to_string(image)
print(text)
