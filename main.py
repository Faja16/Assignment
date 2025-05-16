import customtkinter as ctk
from helper import *
from methods import embed_watermark, extract_watermark, detect_tampering
import os


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Steganography Toolkit")
        self.geometry("600x500")

        # Initialize image paths

        ##FOR EMBEDDING#####
        self.cover_image_path = None
        self.watermark_image_path = None
        self.output_folder_path = None

        ##FOR VERIFYING#####
        self.watermarked_image_path = None
        self.original_watermark_path = None

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
        embedder_btn.grid(pady=10)

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

        try:
            _, output_image_path = embed_watermark(
                self.cover_image_path,
                self.watermark_image_path,
                self.output_folder_path,
            )
            self.embed_status_label.configure(
                text=f"✅ Watermark embedded!\nSaved to:\n{output_image_path}",
                text_color="green",
            )
        except Exception as e:
            self.embed_status_label.configure(
                text=f"❌ Error: {str(e)}", text_color="red"
            )

        def handle_verify(self):
            if not self.watermarked_image_path or not self.original_watermark_path:
                self.embed_status_label.configure(
                    text="❌ Please select all paths.", text_color="red"
                )
                return

            try:
                is_authenticated, extracted_watermarks = extract_watermark(
                    self.watermarked_image_path, self.original_watermark_path
                )
                self.embed_status_label.configure(
                    text=f"✅ Watermark Verified!\nSaved to:\n{output_image_path}",
                    text_color="green",
                )
            except Exception as e:
                self.embed_status_label.configure(
                    text=f"❌ Error: {str(e)}", text_color="red"
                )

    def init_embedder_page(self):
        self.clear_window()

        title = ctk.CTkLabel(self, text="Watermark Embedder", font=("Helvetica", 22))
        title.grid(pady=20)

        # Cover Image Selector
        cover_btn = ctk.CTkButton(
            self,
            text="+ Select Cover Image",
            width=250,
            command=select_image("cover"),
        )
        cover_btn.grid(pady=10)

        # Watermark Image Selector
        watermark_btn = ctk.CTkButton(
            self,
            text="+ Select Watermark Image",
            width=250,
            command=select_image("watermark"),
        )
        watermark_btn.grid(pady=10)

        # Output Folder Selector
        output_btn = ctk.CTkButton(
            self,
            text="Choose Output Folder",
            width=250,
            command=select_folder(),
        )
        output_btn.grid(pady=20)

        # Embed Watermark Button
        embed_btn = ctk.CTkButton(
            self, text="✅ Embed Watermark", width=250, command=self.handle_embed
        )
        embed_btn.grid(pady=20)

        # Label to show feedback
        self.embed_status_label = ctk.CTkLabel(
            self, text="", text_color="green", font=("Arial", 14)
        )
        self.embed_status_label.grid(pady=10)

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
            text="+ Select Image",
            width=250,
            command=select_image("watermarked"),
        )
        watermarked_btn.grid(pady=10)

        # Original Watermark Image Selector
        orig_watermark_btn = ctk.CTkButton(
            self,
            text="+ Select Image",
            width=250,
            command=select_image("original watermark"),
        )
        orig_watermark_btn.grid(pady=10)

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

        # Cover Image Selector
        cover_btn = ctk.CTkButton(
            self,
            text="+ Select Image",
            width=250,
            command=self.select_cover_image,
        )
        cover_btn.grid(pady=10)

        # Back Button
        back_btn = ctk.CTkButton(
            self, text="← Back", width=150, command=self.init_start_page
        )

        back_btn.grid(pady=30)


# Run the app
if __name__ == "__main__":
    app = App()
    app.mainloop()
