import cv2
import customtkinter as ctk
from tkinter import filedialog
from methods import embed_watermark, extract_watermark, detect_tampering
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import threading


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Steganography Toolkit")
        self.geometry("600x1000")

        # Initialize image paths

        ##FOR EMBEDDING#####
        self.cover_image_path = None
        self.watermark_image_path = None
        self.output_folder_path = None

        ##FOR VERIFYING#####
        self.watermarked_image_path = None
        self.original_watermark_path = None

        ####FOR TAMPERING DETECTION#####
        self.subject_image_path = None

        self.init_start_page()

    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

    def init_start_page(self):
        self.clear_window()

        title = ctk.CTkLabel(self, text="Steganography Tool", font=("Helvetica", 24))
        title.grid(pady=40)

        embedder_btn = ctk.CTkButton(
            self, text="Watermark Embedder", width=200, command=self.go_to_embedder
        )
        embedder_btn.grid(pady=10, padx=50)

        verifier_btn = ctk.CTkButton(
            self, text="Authenticity Verifier", width=200, command=self.go_to_verifier
        )
        verifier_btn.grid(pady=10)

        tamper_btn = ctk.CTkButton(
            self, text="Tampering Detector", width=200, command=self.go_to_tampering
        )
        tamper_btn.grid(pady=10)

    def go_to_embedder(self):
        self.init_embedder_page()

    def go_to_verifier(self):
        self.init_verifier_page()

    def go_to_tampering(self):
        self.init_detector_page()

    def handle_embed(self):
        if (
            not self.cover_image_path
            or not self.watermark_image_path
            or not self.output_folder_path
        ):
            self.embed_status_label.configure(
                text="❌ Please select all paths.", text_color="red"
            )
            return

        self.run_embedding(
            self.cover_image_path, self.watermark_image_path, self.output_folder_path
        )

    def handle_verify(self):
        if not self.watermarked_image_path or not self.original_watermark_path:
            self.verify_status_label.configure(
                text="❌ Please select all paths.", text_color="red"
            )
            return

        self.run_extraction(self.watermarked_image_path, self.original_watermark_path)

    def handle_detect(self):
        if not self.subject_image_path or not self.original_watermark_path:
            self.detect_status_label.configure(
                text="❌ Please select all paths.", text_color="red"
            )
            return

        self.run_detection(self.subject_image_path, self.original_watermark_path)

    def init_embedder_page(self):
        self.clear_window()

        title = ctk.CTkLabel(self, text="Watermark Embedder", font=("Helvetica", 22))
        title.grid(pady=20)

        # Cover Image Selector
        cover_btn = ctk.CTkButton(
            self,
            text="+ Select Cover Image",
            width=250,
            command=lambda: self.get_image_path("cover"),
        )
        cover_btn.grid(pady=10)

        self.cover_preview_label = ctk.CTkLabel(self, text="")
        self.cover_preview_label.grid(pady=5)

        # Watermark Image Selector
        watermark_btn = ctk.CTkButton(
            self,
            text="+ Select Watermark Image",
            width=250,
            command=lambda: self.get_image_path("watermark"),
        )
        watermark_btn.grid(pady=10)

        self.watermark_preview_label = ctk.CTkLabel(self, text="")
        self.watermark_preview_label.grid(pady=5)

        # Output Folder Selector
        output_btn = ctk.CTkButton(
            self,
            text="Choose Output Folder",
            width=250,
            command=self.get_folder_path,
        )
        output_btn.grid(pady=20)

        self.output_preview_label = ctk.CTkLabel(self, text="")
        self.output_preview_label.grid(pady=5)

        # Embed Watermark Button
        embed_btn = ctk.CTkButton(
            self,
            text="✅ Embed Watermark",
            width=250,
            command=self.handle_embed,
        )
        embed_btn.grid(pady=20)

        # Label to show feedback
        self.embed_status_label = ctk.CTkLabel(
            self, text="", text_color="green", font=("Arial", 14)
        )
        self.embed_status_label.grid(pady=10)

        self.embed_status_label2 = ctk.CTkLabel(
            self, text="", text_color="green", font=("Arial", 14)
        )
        self.embed_status_label2.grid(pady=10)

        # Back Button
        back_btn = ctk.CTkButton(
            self, text="← Back", width=150, command=self.init_start_page
        )
        back_btn.grid(pady=30)

    def init_verifier_page(self):
        self.clear_window()

        title = ctk.CTkLabel(self, text="Authenticity Verifier", font=("Helvetica", 22))
        title.grid(pady=20)

        # Watermarked Image Selector
        watermarked_btn = ctk.CTkButton(
            self,
            text="+ Select Watermarked Image",
            width=250,
            command=lambda: self.get_image_path("watermarked"),
        )
        watermarked_btn.grid(pady=10)
        self.watermarked_preview_label = ctk.CTkLabel(self, text="")
        self.watermarked_preview_label.grid(pady=5)

        # Original Watermark Image Selector
        orig_watermark_btn = ctk.CTkButton(
            self,
            text="+ Select Original Watermark Image",
            width=250,
            command=lambda: self.get_image_path("original watermark"),
        )
        orig_watermark_btn.grid(pady=10)
        self.orig_watermark_preview_label = ctk.CTkLabel(self, text="")
        self.orig_watermark_preview_label.grid(pady=5)

        # Verify Auth Button
        verify_btn = ctk.CTkButton(
            self,
            text="Verify Watermark",
            width=250,
            command=self.handle_verify,
        )
        verify_btn.grid(pady=20)

        # Label to show feedback
        self.verify_status_label = ctk.CTkLabel(
            self, text="", text_color="green", font=("Arial", 14)
        )
        self.verify_status_label.grid(pady=10)

        # Back Button
        back_btn = ctk.CTkButton(
            self, text="← Back", width=150, command=self.init_start_page
        )
        back_btn.grid(pady=30)

    def init_detector_page(self):
        self.clear_window()

        title = ctk.CTkLabel(self, text="Tampering Detector", font=("Helvetica", 22))
        title.grid(pady=20)

        # Subject Image Selector
        sub_image_btn = ctk.CTkButton(
            self,
            text="+ Select Subject Image",
            width=250,
            command=lambda: self.get_image_path("subject image"),
        )
        sub_image_btn.grid(pady=10)
        self.sub_image_preview_label = ctk.CTkLabel(self, text="")
        self.sub_image_preview_label.grid(pady=5)

        # Original Watermark Image Selector
        orig_watermark_btn = ctk.CTkButton(
            self,
            text="+ Select Original Watermark Image",
            width=250,
            command=lambda: self.get_image_path("original watermark"),
        )
        orig_watermark_btn.grid(pady=10)
        self.orig_watermark_preview_label = ctk.CTkLabel(self, text="")
        self.orig_watermark_preview_label.grid(pady=5)

        # Detect Tampering Button
        detct_btn = ctk.CTkButton(
            self,
            text="Detect Tampering",
            width=250,
            command=self.handle_detect,
        )
        detct_btn.grid(pady=20)

        # Label to show feedback
        self.detect_status_label = ctk.CTkLabel(
            self, text="", text_color="green", font=("Arial", 14)
        )
        self.detect_status_label.grid(pady=10)

        # Back Button
        back_btn = ctk.CTkButton(
            self, text="← Back", width=150, command=self.init_start_page
        )

        back_btn.grid(pady=30)

    def get_image_path(self, name):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not path:
            return  # Handle cancel case

        img = Image.open(path).resize((150, 150))
        img = ImageTk.PhotoImage(img)  # Convert for Tkinter

        if name == "cover":
            self.cover_image_path = path
            self.cover_preview_label.configure(image=img, text="")
            self.cover_preview_label.image = img  # Keep reference

        elif name == "watermark":
            self.watermark_image_path = path
            self.watermark_preview_label.configure(image=img, text="")
            self.watermark_preview_label.image = img

        elif name == "watermarked":
            self.watermarked_image_path = path
            self.watermarked_preview_label.configure(image=img, text="")
            self.watermarked_preview_label.image = img

        elif name == "original watermark":
            self.original_watermark_path = path
            self.orig_watermark_preview_label.configure(image=img, text="")
            self.orig_watermark_preview_label.image = img

        elif name == "subject image":
            self.subject_image_path = path
            self.sub_image_preview_label.configure(image=img, text="")
            self.sub_image_preview_label.image = img

    def get_folder_path(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder_path = folder
            self.output_preview_label.configure(
                text=f"Folder: {os.path.basename(folder)}"
            )
            print("Output folder selected:", folder)
        else:
            print("Please try again")

    def run_detection(self, subject_image_path, original_watermark_path):
        self.detect_status_label.configure(
            text="Decting Alterations...", text_color="orange"
        )
        app.update_idletasks()

        def task():
            try:
                is_tampered, tampered_image = detect_tampering(
                    subject_image_path, original_watermark_path
                )

                if is_tampered:

                    self.detect_status_label.configure(
                        text=f"❌ Image has been tampered with!",
                        text_color="red",
                    )

                    plt.imshow(tampered_image)

                    plt.show()
                else:
                    self.detect_status_label.configure(
                        text=f"✅ No tampering has been detected!",
                        text_color="green",
                    )
            except Exception as e:
                self.detect_status_label.configure(
                    text=f"❌ Error: {str(e)}", text_color="red"
                )
                print(str(e))

        threading.Thread(target=task).start()

    def run_embedding(self, cover_image_path, watermark_image_path, output_folder_path):
        self.verify_status_label.configure(text="Embedding...", text_color="orange")
        app.update_idletasks()

        def task():
            try:
                _, output_image_path = embed_watermark(
                    cover_image_path,
                    watermark_image_path,
                    output_folder_path,
                )

                img = Image.open(output_image_path).resize((150, 150))
                img = ImageTk.PhotoImage(img)  # Convert for Tkinter

                self.embed_status_label.configure(image=img)
                self.embed_status_label.image = img

                self.embed_status_label2.configure(
                    text=f"✅ Watermark embedded!\nSaved to:\n{output_image_path}",
                    text_color="green",
                )
            except Exception as e:
                self.embed_status_label.configure(
                    text=f"❌ Error: {str(e)}", text_color="red"
                )
                print(str(e))

        threading.Thread(target=task).start()

    def run_extraction(self, watermarked_image_path, original_watermark_path):

        self.verify_status_label.configure(
            text="Authenticating...", text_color="orange"
        )
        app.update_idletasks()

        def task():
            try:
                is_authenticated, _ = extract_watermark(
                    watermarked_image_path, original_watermark_path
                )

                if is_authenticated:

                    self.verify_status_label.configure(
                        text=f"✅ Watermark Verified!",
                        text_color="green",
                    )
                else:
                    self.verify_status_label.configure(
                        text=f"❌ Watermark Not Found!",
                        text_color="red",
                    )
            except Exception as e:
                self.verify_status_label.configure(
                    text=f"❌ Error: {str(e)}", text_color="red"
                )
                print(str(e))

        threading.Thread(target=task).start()


# Run the app
if __name__ == "__main__":
    app = App()
    app.grid_columnconfigure(0, weight=1)
    app.mainloop()
