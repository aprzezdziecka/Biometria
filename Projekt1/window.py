import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

from part1 import (
    read_image, to_array, to_image, to_grayscale_simple, to_grayscale, 
    brightness, gamma_correction, contrast_correction, negative, 
    binarization, apply_kernel, mean_kernel, gaussian_kernel, 
    sharpening_kernel, histogram, line_rgb, vertical_projection, 
    horizontal_projection, Roberts_cross, Sobel, any_filter, 
    otsu_binarization, kuwahara_filter, wave_distortion,
    median_filter, Prewitt, Laplace, add_salt_and_pepper 
)

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Edytor Obrazów")
        
        # Zmienne przechowujące stan obrazu
        self.original_image = None
        self.tk_original = None
        self.tk_processed = None
        self.custom_kernel_matrix = None
        self.is_processed_gray = False

        self.label_info = tk.Label(self.root, text="Rozdzielczość: --- x --- px", fg="blue")
        self.label_info.pack(pady=5)
        
        self.setup_ui()

    def setup_ui(self):
        # --- GŁÓWNE RAMKI ---
        self.menu_frame = tk.Frame(self.root, pady=10)
        self.menu_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.controls_frame = tk.Frame(self.root, padx=10, pady=10)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.images_frame = tk.Frame(self.root, padx=10, pady=10)
        self.images_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # --- MENU ---
        tk.Button(self.menu_frame, text="Wczytaj Obraz", command=self.load_image, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(self.menu_frame, text="Zapisz Obraz", command=self.save_image, width=15).pack(side=tk.LEFT, padx=5)
        
        self.btn_apply = tk.Button(self.menu_frame, text="ZASTOSUJ FILTRY", command=self.process_image, width=20, bg="green", fg="white", font=("Arial", 10, "bold"))
        self.btn_apply.pack(side=tk.RIGHT, padx=20)

        self.lbl_status = tk.Label(self.menu_frame, text="", fg="red", font=("Arial", 10, "italic"))
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # --- PODGLĄD OBRAZÓW ---
        self.lbl_original = tk.Label(self.images_frame, text="Obraz oryginalny", bg="gray", width=50, height=20)
        self.lbl_original.pack(side=tk.LEFT, padx=10, expand=True)
        
        self.lbl_processed = tk.Label(self.images_frame, text="Obraz po zmianach", bg="gray", width=50, height=20)
        self.lbl_processed.pack(side=tk.RIGHT, padx=10, expand=True)

        # --- KONTROLKI FILTRÓW (CHECKBOXY I PARAMETRY) ---
        self.var_gray1 = tk.IntVar()
        self.var_gray2 = tk.IntVar()
        self.var_negative = tk.IntVar()
        self.var_brightness = tk.IntVar()
        self.var_contrast = tk.IntVar()
        self.var_gamma = tk.IntVar()
        self.var_blur_mean = tk.IntVar()
        self.var_blur_gauss = tk.IntVar()
        self.var_sharpen = tk.IntVar()
        self.var_custom = tk.IntVar()
        self.var_median = tk.IntVar()
        self.var_kuwahara = tk.IntVar()
        self.var_wave = tk.IntVar()
        self.var_noise = tk.IntVar()
        self.var_binary = tk.IntVar()
        self.var_otsu = tk.IntVar()
        self.var_sobel = tk.IntVar()
        self.var_roberts = tk.IntVar()
        self.var_prewitt = tk.IntVar()
        self.var_laplace = tk.IntVar()

        # Sekcja 1: Podstawowe (Kolor)
        tk.Label(self.controls_frame, text="--- Podstawowe ---", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=(0,5))
        tk.Checkbutton(self.controls_frame, text="Skala szarości (Suma)", variable=self.var_gray1, command=self.update_ui_state).grid(row=1, column=0, sticky=tk.W)
        tk.Checkbutton(self.controls_frame, text="Skala szarości (Wagi)", variable=self.var_gray2, command=self.update_ui_state).grid(row=2, column=0, sticky=tk.W)
        tk.Checkbutton(self.controls_frame, text="Negatyw", variable=self.var_negative).grid(row=3, column=0, sticky=tk.W)

        # Sekcja 2: Parametryczne
        tk.Label(self.controls_frame, text="--- Korekcja ---", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2, pady=(10,5))
        
        tk.Checkbutton(self.controls_frame, text="Jasność (b):", variable=self.var_brightness).grid(row=5, column=0, sticky=tk.W)
        self.ent_brightness = tk.Entry(self.controls_frame, width=8); self.ent_brightness.grid(row=5, column=1); self.ent_brightness.insert(0, "50")

        tk.Checkbutton(self.controls_frame, text="Kontrast (c):", variable=self.var_contrast).grid(row=6, column=0, sticky=tk.W)
        self.ent_contrast = tk.Entry(self.controls_frame, width=8); self.ent_contrast.grid(row=6, column=1); self.ent_contrast.insert(0, "1.5")

        tk.Checkbutton(self.controls_frame, text="Gamma:", variable=self.var_gamma).grid(row=7, column=0, sticky=tk.W)
        self.ent_gamma = tk.Entry(self.controls_frame, width=8); self.ent_gamma.grid(row=7, column=1); self.ent_gamma.insert(0, "1.2")

        # Sekcja 3: Filtry splotowe i przestrzenne
        tk.Label(self.controls_frame, text="--- Filtry i Sploty ---", font=("Arial", 10, "bold")).grid(row=8, column=0, columnspan=2, pady=(10,5))
        
        tk.Checkbutton(self.controls_frame, text="Rozmycie Średnie (rozmiar):", variable=self.var_blur_mean).grid(row=9, column=0, sticky=tk.W)
        self.ent_mean_size = tk.Entry(self.controls_frame, width=8); self.ent_mean_size.grid(row=9, column=1); self.ent_mean_size.insert(0, "3")

        tk.Checkbutton(self.controls_frame, text="Rozmycie Gaussa (rozmiar, sigma):", variable=self.var_blur_gauss).grid(row=10, column=0, sticky=tk.W)
        self.ent_gauss_params = tk.Entry(self.controls_frame, width=8); self.ent_gauss_params.grid(row=10, column=1); self.ent_gauss_params.insert(0, "3, 1.0")

        tk.Checkbutton(self.controls_frame, text="Wyostrzanie (rozmiar, ilość):", variable=self.var_sharpen).grid(row=11, column=0, sticky=tk.W)
        self.ent_sharpen_params = tk.Entry(self.controls_frame, width=8); self.ent_sharpen_params.grid(row=11, column=1); self.ent_sharpen_params.insert(0, "3, 1.5")

        tk.Checkbutton(self.controls_frame, text="Własny Kernel:", variable=self.var_custom).grid(row=12, column=0, sticky=tk.W)
        tk.Button(self.controls_frame, text="Stwórz Kernel", command=self.open_kernel_window).grid(row=12, column=1)

        tk.Checkbutton(self.controls_frame, text="Filtr Medianowy (rozmiar):", variable=self.var_median).grid(row=13, column=0, sticky=tk.W)
        self.ent_median = tk.Entry(self.controls_frame, width=8); self.ent_median.grid(row=13, column=1); self.ent_median.insert(0, "3")

        tk.Checkbutton(self.controls_frame, text="Filtr Kuwahary (rozmiar):", variable=self.var_kuwahara).grid(row=14, column=0, sticky=tk.W)
        self.ent_kuwahara = tk.Entry(self.controls_frame, width=8); self.ent_kuwahara.grid(row=14, column=1); self.ent_kuwahara.insert(0, "5")

        tk.Checkbutton(self.controls_frame, text="Dodaj Szum (gęstość):", variable=self.var_noise).grid(row=15, column=0, sticky=tk.W)
        self.ent_noise = tk.Entry(self.controls_frame, width=8); self.ent_noise.grid(row=15, column=1); self.ent_noise.insert(0, "0.05")

        tk.Checkbutton(self.controls_frame, text="Zakrzywienie Fali (amp, dług):", variable=self.var_wave).grid(row=16, column=0, sticky=tk.W)
        self.ent_wave = tk.Entry(self.controls_frame, width=8); self.ent_wave.grid(row=16, column=1); self.ent_wave.insert(0, "20, 100")

        # Sekcja 4: Wymagające skali szarości
        tk.Label(self.controls_frame, text="--- Tylko dla szarości ---", font=("Arial", 10, "bold")).grid(row=17, column=0, columnspan=2, pady=(10,5))
        
        self.cb_binary = tk.Checkbutton(self.controls_frame, text="Binaryzacja Ręczna (próg):", variable=self.var_binary)
        self.cb_binary.grid(row=18, column=0, sticky=tk.W)
        self.ent_binary = tk.Entry(self.controls_frame, width=8); self.ent_binary.grid(row=18, column=1); self.ent_binary.insert(0, "128")

        self.cb_otsu = tk.Checkbutton(self.controls_frame, text="Binaryzacja Otsu (Auto)", variable=self.var_otsu)
        self.cb_otsu.grid(row=19, column=0, sticky=tk.W, columnspan=2)

        self.cb_sobel = tk.Checkbutton(self.controls_frame, text="Detekcja Krawędzi (Sobel)", variable=self.var_sobel)
        self.cb_sobel.grid(row=20, column=0, sticky=tk.W, columnspan=2)
        
        self.cb_roberts = tk.Checkbutton(self.controls_frame, text="Detekcja Krawędzi (Roberts)", variable=self.var_roberts)
        self.cb_roberts.grid(row=21, column=0, sticky=tk.W, columnspan=2)

        self.cb_prewitt = tk.Checkbutton(self.controls_frame, text="Detekcja Krawędzi (Prewitt)", variable=self.var_prewitt)
        self.cb_prewitt.grid(row=22, column=0, sticky=tk.W, columnspan=2)

        self.cb_laplace = tk.Checkbutton(self.controls_frame, text="Detekcja Krawędzi (Laplace)", variable=self.var_laplace)
        self.cb_laplace.grid(row=23, column=0, sticky=tk.W, columnspan=2)

        # Sekcja 5: Analiza (Wykresy)
        tk.Label(self.controls_frame, text="--- Analiza Obrazu ---", font=("Arial", 10, "bold")).grid(row=24, column=0, columnspan=2, pady=(10,5))
        
        analiza_frame = tk.Frame(self.controls_frame)
        analiza_frame.grid(row=25, column=0, columnspan=2)
        
        tk.Button(analiza_frame, text="Histogram RGB/Szary", command=self.show_histogram).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(analiza_frame, text="Linie RGB", command=self.show_line_rgb).grid(row=0, column=1, padx=2, pady=2)
        
        self.btn_proj_v = tk.Button(analiza_frame, text="Projekcja Pionowa", command=self.show_proj_v)
        self.btn_proj_v.grid(row=1, column=0, padx=2, pady=2)
        self.btn_proj_h = tk.Button(analiza_frame, text="Projekcja Pozioma", command=self.show_proj_h)
        self.btn_proj_h.grid(row=1, column=1, padx=2, pady=2)

        self.update_ui_state()

    def update_ui_state(self):
        """Blokuje lub odblokowuje widżety zależne od skali szarości."""
        is_gray = self.var_gray1.get() == 1 or self.var_gray2.get() == 1
        state = tk.NORMAL if getattr(self, 'is_processed_gray', False) else tk.DISABLED
        
        self.cb_binary.config(state=state)
        self.ent_binary.config(state=state)
        self.cb_otsu.config(state=state)
        self.cb_sobel.config(state=state)
        self.cb_roberts.config(state=state)
        self.cb_prewitt.config(state=state)
        self.cb_laplace.config(state=state)
        self.btn_proj_v.config(state=state)
        self.btn_proj_h.config(state=state)

        if not is_gray:
            self.var_binary.set(0)
            self.var_otsu.set(0)
            self.var_sobel.set(0)
            self.var_roberts.set(0)
            self.var_prewitt.set(0)
            self.var_laplace.set(0)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Pliki obrazów", "*.png *.jpg *.jpeg *.bmp"), ("Wszystkie", "*.*")]
        )
        if file_path:
            try:
                self.original_image = read_image(file_path)
                if self.original_image.mode == 'L':
                    self.is_processed_gray = True
                    self.var_gray1.set(1)
                else:
                    self.is_processed_gray = False
                    self.var_gray1.set(0)
                self.update_ui_state()
                width, height = self.original_image.size
                self.label_info.config(text=f"Rozdzielczość: {width} x {height} px")
                display_img = self.original_image.copy()
                display_img.thumbnail((400, 400))
                self.tk_original = ImageTk.PhotoImage(display_img)
                self.lbl_original.config(image=self.tk_original, text="", width=display_img.width, height=display_img.height)
                
                self.lbl_processed.config(image='', text="Czeka na filtry...")
                self.custom_kernel_matrix = None
                self.var_custom.set(0)
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się wczytać obrazu: {e}")

    def save_image(self):
        if not hasattr(self, 'current_processed_pil') or self.current_processed_pil is None:
            messagebox.showwarning("Uwaga", "Brak przetworzonego obrazu do zapisania!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Wszystkie", "*.*")]
        )
        if file_path:
            try:
                self.current_processed_pil.save(file_path)
                messagebox.showinfo("Sukces", "Obraz zapisany pomyślnie!")
            except Exception as e:
                messagebox.showerror("Błąd", f"Błąd zapisu: {e}")

    def process_image(self):
        """Uruchamia proces w nowym wątku, aby nie blokować interfejsu."""
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz!")
            return

        # Zablokuj przycisk i poinformuj użytkownika
        self.btn_apply.config(state=tk.DISABLED, bg="gray")
        self.lbl_status.config(text="Przetwarzanie w tle... Proszę czekać.", fg="red")
        self.lbl_processed.config(image='', text="Trwają obliczenia...")

        # Uruchomienie wątku tła
        thread = threading.Thread(target=self._process_image_background, daemon=True)
        thread.start()

    def _process_image_background(self):
        """Logika nakładania filtrów (działa poza głównym wątkiem GUI)."""
        try:
            start_time = time.time()
            img_array = to_array(self.original_image)
            is_currently_gray = False

            if self.var_gray1.get() == 1:
                img_array = to_grayscale_simple(img_array)
                is_currently_gray = True
            elif self.var_gray2.get() == 1:
                img_array = to_grayscale(img_array)
                is_currently_gray = True

            if self.var_negative.get() == 1:
                img_array = negative(img_array)

            if self.var_brightness.get() == 1:
                b = float(self.ent_brightness.get())
                img_array = brightness(img_array, b)

            if self.var_contrast.get() == 1:
                c = float(self.ent_contrast.get())
                img_array = contrast_correction(img_array, c)

            if self.var_gamma.get() == 1:
                gamma = float(self.ent_gamma.get())
                img_array = gamma_correction(img_array, gamma)

            if self.var_blur_mean.get() == 1:
                size = int(self.ent_mean_size.get())
                kernel = mean_kernel(size)
                img_array = apply_kernel(img_array, kernel)

            if self.var_blur_gauss.get() == 1:
                params = self.ent_gauss_params.get().split(',')
                size, sigma = int(params[0].strip()), float(params[1].strip())
                kernel = gaussian_kernel(size, sigma)
                img_array = apply_kernel(img_array, kernel)

            if self.var_sharpen.get() == 1:
                params = self.ent_sharpen_params.get().split(',')
                size, amount = int(params[0].strip()), float(params[1].strip())
                kernel = sharpening_kernel(size, amount)
                img_array = apply_kernel(img_array, kernel)

            if self.var_custom.get() == 1 and self.custom_kernel_matrix is not None:
                img_array = any_filter(img_array, self.custom_kernel_matrix)

            if self.var_median.get() == 1:
                size = int(self.ent_median.get())
                img_array = median_filter(img_array, size)

            if self.var_kuwahara.get() == 1:
                size = int(self.ent_kuwahara.get())
                img_array = kuwahara_filter(img_array, size)

            if self.var_noise.get() == 1:
                density = float(self.ent_noise.get())
                img_array = add_salt_and_pepper(img_array, density)
                
            if self.var_wave.get() == 1:
                params = self.ent_wave.get().split(',')
                amp, wave_len = int(params[0].strip()), int(params[1].strip())
                img_array = wave_distortion(img_array, amp, wave_len)

            if is_currently_gray:
                if self.var_binary.get() == 1:
                    threshold = float(self.ent_binary.get())
                    img_array = binarization(img_array, threshold)
                
                if self.var_otsu.get() == 1:
                    img_array = otsu_binarization(img_array)
                
                if self.var_sobel.get() == 1:
                    img_array = Sobel(img_array)
                
                if self.var_roberts.get() == 1:
                    img_array = Roberts_cross(img_array)
                    
                if self.var_prewitt.get() == 1:
                    img_array = Prewitt(img_array)
                    
                if self.var_laplace.get() == 1:
                    img_array = Laplace(img_array)

            self.current_array_state = img_array
            self.current_processed_pil = to_image(img_array)

            end_time = time.time()
            elapsed_time = end_time - start_time

            self.root.after(0, self._finalize_processing, True, None, elapsed_time)

        except ValueError as e:
            self.root.after(0, self._finalize_processing, False, "Błąd wartości: upewnij się, że wpisane parametry są poprawne.\n" + str(e))
        except Exception as e:
            self.root.after(0, self._finalize_processing, False, str(e), 0)

    def _finalize_processing(self, success, error_msg, elapsed_time):
        """Aktualizacja UI z powrotem w głównym wątku po zakończeniu obliczeń."""
        self.btn_apply.config(state=tk.NORMAL, bg="green")
        
        if success:
            self.is_processed_gray = (self.current_array_state.ndim == 2)
            self.update_ui_state()

            status_text = f"Gotowe w {elapsed_time:.2f} s!" 
            self.lbl_status.config(text=status_text, fg="green")
            
            display_img = self.current_processed_pil.copy()
            display_img.thumbnail((400, 400))
            self.tk_processed = ImageTk.PhotoImage(display_img)
            self.lbl_processed.config(image=self.tk_processed, text="", width=display_img.width, height=display_img.height)
        else:
            self.lbl_status.config(text="Wystąpił błąd", fg="red")
            self.lbl_processed.config(text="Błąd przetwarzania")
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas nakładania filtrów:\n{error_msg}")

    # --- OKNO KERNELA ---
    def open_kernel_window(self):
        kernel_window = tk.Toplevel(self.root)
        kernel_window.title("Kreator Kernela")
        kernel_window.geometry("300x300")
        
        tk.Label(kernel_window, text="Wymiar N (nieparzysty, np. 3):").pack(pady=5)
        ent_size = tk.Entry(kernel_window)
        ent_size.pack()
        
        grid_frame = tk.Frame(kernel_window)
        grid_frame.pack(pady=10)
        
        entries = []

        def generate_grid():
            for widget in grid_frame.winfo_children():
                widget.destroy()
            entries.clear()
            try:
                size = int(ent_size.get())
                if size % 2 == 0:
                    messagebox.showwarning("Błąd", "Wymiar musi być liczbą nieparzystą!")
                    return
                for i in range(size):
                    row_entries = []
                    for j in range(size):
                        e = tk.Entry(grid_frame, width=5)
                        e.grid(row=i, column=j, padx=2, pady=2)
                        e.insert(0, "0")
                        row_entries.append(e)
                    entries.append(row_entries)
            except ValueError:
                messagebox.showerror("Błąd", "Wprowadź prawidłową liczbę całkowitą.")

        def save_kernel():
            if not entries: return
            try:
                size = len(entries)
                matrix = np.zeros((size, size))
                for i in range(size):
                    for j in range(size):
                        matrix[i, j] = float(entries[i][j].get())
                self.custom_kernel_matrix = matrix
                self.var_custom.set(1) 
                messagebox.showinfo("Sukces", "Kernel został zapisany w pamięci.")
                kernel_window.destroy()
            except ValueError:
                messagebox.showerror("Błąd", "Upewnij się, że wszystkie pola kernela są liczbami.")

        tk.Button(kernel_window, text="Generuj Siatkę", command=generate_grid).pack(pady=5)
        tk.Button(kernel_window, text="Zatwierdź Kernel", command=save_kernel, bg="blue", fg="white").pack(side=tk.BOTTOM, pady=10)

    # --- FUNKCJE ANALITYCZNE ---
    def get_analysis_array(self):
        if hasattr(self, 'current_array_state'):
            return self.current_array_state
        elif self.original_image:
            return to_array(self.original_image)
        else:
            messagebox.showwarning("Brak", "Wczytaj obraz!")
            return None

    def show_histogram(self):
        arr = self.get_analysis_array()
        if arr is not None:
            plt.close('all') 
            plt.figure(figsize=(8, 6)) 
            histogram(arr)
            plt.show()

    def show_line_rgb(self):
        arr = self.get_analysis_array()
        if arr is not None:
            if arr.ndim != 3:
                messagebox.showinfo("Informacja", "Ten obraz nie ma formatu RGB (jest szary).")
                return
            plt.close('all') 
            line_rgb(arr)
            plt.show()

    def show_proj_v(self):
        arr = self.get_analysis_array()
        if arr is not None:
            plt.close('all') 
            vertical_projection(arr)
            plt.show()

    def show_proj_h(self):
        arr = self.get_analysis_array()
        if arr is not None:
            plt.close('all') 
            horizontal_projection(arr)
            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()