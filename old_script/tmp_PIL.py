from PIL import Image


im = Image.open("/home/zj/Desktop/fore_0.jpg")
im = im.resize((50,50))
im.show()