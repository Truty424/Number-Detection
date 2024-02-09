from PIL import Image

# otwórz obraz
image = Image.open("numberPng.png")

# usuń niepotrzebną przestrzeń
image = image.crop(image.getbbox())

# uzyskaj wymiary obrazu
width, height = image.size

# uzyskaj wymiary liczby
number_width, number_height = (width // 2), (height // 2)

# wyznacz punkt środkowy
center = (width // 2, height // 2)

# wyznacz punkt środkowy liczby
number_center = (center[0] - (number_width // 2), center[1] - (number_height // 2))

# wstaw liczbę na środku
image.paste(image.crop((number_center[0], number_center[1], number_center[0]+number_width, number_center[1]+number_height)), center)

# zapisz zmodyfikowany obraz
image.save("nowa_nazwa_pliku.png")
