import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import time

from models.unet_attention import UNetAttention
from utils.data_utils import poisson_blend

class SelectiveSharpeningApp:
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.image_path = None
        self.image = None
        self.mask = None
        self.result = None
        self.drawing = False
        self.brush_size = 15
        self.eraser_mode = False
        self.last_x, self.last_y = None, None
        
        # Try to load model (but don't block UI)
        self.model = None
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Setup UI
        self.setup_ui()
        
    def load_model(self):
        """Load the model in a separate thread to avoid blocking the UI"""
        try:
            # Check if CUDA is available
            if self.device == 'cuda' and not torch.cuda.is_available():
                print("CUDA not available, using CPU instead.")
                self.device = 'cpu'
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Initialize model
            self.model = UNetAttention(n_channels=3, n_classes=3, with_mask_channel=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Update UI
            self.root.after(0, lambda: self.status_var.set(
                f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})"
            ))
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.root.after(0, lambda: self.status_var.set(f"Error loading model: {e}"))
            self.model = None
            
    def setup_ui(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Selective Image Sharpening")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Use a theme if available
        try:
            self.root.tk.call("source", "azure.tcl")
            self.root.tk.call("set_theme", "light")
        except:
            pass
        
        # Create main layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for controls
        left_panel = ttk.Frame(main_frame, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Create right panel for image display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control sections with labels
        file_section = ttk.LabelFrame(left_panel, text="File Operations")
        file_section.pack(fill=tk.X, pady=5)
        
        drawing_section = ttk.LabelFrame(left_panel, text="Drawing Tools")
        drawing_section.pack(fill=tk.X, pady=5)
        
        processing_section = ttk.LabelFrame(left_panel, text="Processing")
        processing_section.pack(fill=tk.X, pady=5)
        
        # Add buttons to file section
        ttk.Button(file_section, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(file_section, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=5, padx=5)
        
        # Add drawing controls to drawing section
        ttk.Button(drawing_section, text="Clear Mask", command=self.clear_mask).pack(fill=tk.X, pady=5, padx=5)
        
        # Drawing mode radio buttons
        self.drawing_mode = tk.StringVar(value="brush")
        ttk.Radiobutton(drawing_section, text="Brush (Draw Mask)", variable=self.drawing_mode, 
                       value="brush", command=self.set_brush_mode).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(drawing_section, text="Eraser (Remove Mask)", variable=self.drawing_mode, 
                       value="eraser", command=self.set_eraser_mode).pack(anchor=tk.W, padx=5)
        
        # Add brush size control
        ttk.Label(drawing_section, text="Brush Size:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        self.brush_slider = ttk.Scale(drawing_section, from_=1, to=50, orient=tk.HORIZONTAL, 
                                     command=self.update_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(fill=tk.X, padx=5, pady=5)
        
        # Add processing button to processing section
        ttk.Button(processing_section, text="Process Image", command=self.process_image).pack(fill=tk.X, pady=5, padx=5)
        
        # Add progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(processing_section, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5, padx=5)
        
        # Create figure for image display
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.fig.subplots_adjust(wspace=0.05, hspace=0)
        
        # Initialize empty plots
        for ax in self.axes:
            ax.axis('off')
        
        self.axes[0].set_title("Original Image\nClick and drag to draw mask")
        self.axes[1].set_title("Mask")
        self.axes[2].set_title("Result")
        
        # Create canvas for matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        self.toolbar.update()
        
        # Connect events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Load an image to start.")
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=2)
        
    def set_brush_mode(self):
        """Set drawing mode to brush"""
        self.eraser_mode = False
        self.status_var.set("Brush mode: Drawing mask")
        
    def set_eraser_mode(self):
        """Set drawing mode to eraser"""
        self.eraser_mode = True
        self.status_var.set("Eraser mode: Removing mask")
        
    def load_image(self):
        """Load an image from file"""
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not self.image_path:
            return
            
        try:
            # Load image
            self.image = Image.open(self.image_path).convert('RGB')
            
            # Resize if too large
            width, height = self.image.size
            max_size = 800
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                self.image = self.image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create empty mask
            self.mask = Image.new('L', self.image.size, 0)
            self.mask_draw = ImageDraw.Draw(self.mask)
            
            # Clear previous result
            self.result = None
            
            # Update display
            self.update_display()
            
            self.status_var.set(f"Loaded image: {os.path.basename(self.image_path)} ({self.image.width}x{self.image.height})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            
    def clear_mask(self):
        """Clear the mask"""
        if self.mask:
            self.mask = Image.new('L', self.image.size, 0)
            self.mask_draw = ImageDraw.Draw(self.mask)
            self.update_display()
            self.status_var.set("Mask cleared")
            
    def update_brush_size(self, value):
        """Update the brush size"""
        self.brush_size = int(float(value))
            
    def on_press(self, event):
        """Handle mouse press event"""
        if not self.image or event.inaxes != self.axes[0]:
            return
            
        self.drawing = True
        self.last_x = int(event.xdata)
        self.last_y = int(event.ydata)
            
    def on_release(self, event):
        """Handle mouse release event"""
        self.drawing = False
        self.last_x, self.last_y = None, None
            
    def on_motion(self, event):
        """Handle mouse motion event"""
        if not self.drawing or not self.image or event.inaxes != self.axes[0]:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Draw line between last position and current position
        if self.last_x is not None and self.last_y is not None:
            fill_value = 0 if self.eraser_mode else 255
            self.mask_draw.line([(self.last_x, self.last_y), (x, y)], fill=fill_value, width=self.brush_size)
        else:
            # Draw single point
            fill_value = 0 if self.eraser_mode else 255
            self.mask_draw.ellipse(
                [(x - self.brush_size // 2, y - self.brush_size // 2),
                 (x + self.brush_size // 2, y + self.brush_size // 2)],
                fill=fill_value
            )
            
        self.last_x, self.last_y = x, y
        
        # Update display
        self.update_display()
            
    def update_display(self):
        """Update the displayed images"""
        # Clear axes
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
            
        # Display original image
        if self.image:
            self.axes[0].imshow(np.array(self.image))
            self.axes[0].set_title("Original Image\nClick and drag to draw mask")
            
        # Display mask
        if self.mask:
            self.axes[1].imshow(np.array(self.mask), cmap='gray')
            self.axes[1].set_title("Mask")
            
        # Display result
        if self.result is not None:
            self.axes[2].imshow(self.result)
            self.axes[2].set_title("Sharpened Result")
            
        # Update canvas
        self.canvas.draw()
            
    def process_image(self):
        """Process the image using the model"""
        if not self.image:
            messagebox.showinfo("Info", "Please load an image first.")
            return
            
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please wait for model to load or check console for details.")
            return
            
        # Check if mask is empty
        mask_array = np.array(self.mask)
        if mask_array.max() == 0:
            messagebox.showinfo("Info", "Please draw a mask first.")
            return
            
        # Update status
        self.status_var.set("Processing image...")
        self.progress_var.set(10)
        
        # Create a thread to avoid freezing the UI
        threading.Thread(target=self._process_in_thread).start()
            
    def _process_in_thread(self):
        """Process the image in a separate thread"""
        try:
            # Convert to tensors
            self.root.after(0, lambda: self.progress_var.set(20))
            transform = transforms.ToTensor()
            input_tensor = transform(self.image).unsqueeze(0)
            mask_tensor = transform(self.mask).unsqueeze(0)
            
            # Move to device
            self.root.after(0, lambda: self.progress_var.set(40))
            input_tensor = input_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)
            
            # Apply model
            self.root.after(0, lambda: self.progress_var.set(60))
            self.root.after(0, lambda: self.status_var.set("Running model..."))
            with torch.no_grad():
                output = self.model(input_tensor, mask_tensor)
                
            # Convert to numpy for processing
            self.root.after(0, lambda: self.progress_var.set(80))
            self.root.after(0, lambda: self.status_var.set("Applying Poisson blending..."))
            input_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()
            output_np = output[0].cpu().permute(1, 2, 0).numpy()
            mask_np = mask_tensor[0].cpu().squeeze().numpy()
            
            # Apply Poisson blending
            self.result = poisson_blend(input_np, output_np, mask_np)
            
            # Update UI
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, self._update_after_processing)
        except Exception as e:
            print(f"Error during processing: {e}")
            self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
            self.root.after(0, lambda: self.progress_var.set(0))
            
    def _update_after_processing(self):
        """Update UI after processing is complete"""
        self.update_display()
        self.status_var.set("Processing complete")
        # Reset progress bar after a delay
        self.root.after(1000, lambda: self.progress_var.set(0))
            
    def save_result(self):
        """Save the result to file"""
        if self.result is None:
            messagebox.showinfo("Info", "No result to save. Please process an image first.")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        
        if not save_path:
            return
            
        try:
            # Convert result to image and save
            result_img = Image.fromarray((self.result * 255).astype(np.uint8))
            result_img.save(save_path)
            
            # Save mask as well
            if messagebox.askyesno("Save Mask", "Would you like to save the mask as well?"):
                mask_path = save_path.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
                self.mask.save(mask_path)
                self.status_var.set(f"Result saved to {os.path.basename(save_path)} and mask saved to {os.path.basename(mask_path)}")
            else:
                self.status_var.set(f"Result saved to {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result: {e}")
            
    def run(self):
        """Run the application"""
        self.root.mainloop()

def parse_args():
    parser = argparse.ArgumentParser(description='Selective Image Sharpening Demo')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def main(args=None):
    # Parse arguments if not provided
    if args is None:
        args = parse_args()
    
    # Use CPU if CUDA is not available
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available, using CPU instead.")
        device = 'cpu'
    else:
        device = args.device
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Warning: Model file not found at {args.model_path}")
        print("The application will start, but you will need to provide a valid model to process images.")
        
    # Start the application
    app = SelectiveSharpeningApp(args.model_path, device)
    app.run()

if __name__ == '__main__':
    main() 