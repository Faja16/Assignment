from tkinter import filedialog


def select_image(name):
    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
    )
    if path:
        image_path = path
        print(f"{name}image selected:", path)
    else:
        print("Please try again")

    return image_path


def select_folder():
    folder = filedialog.askdirectory()
    if folder:
        output_folder_path = folder
        print("Output folder selected:", folder)
    else:
        print("Please try again")

    return output_folder_path
