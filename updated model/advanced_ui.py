#!/usr/bin/env python
"""
Advanced User Interface for Selective Image Sharpening

This application provides an integrated interface for all the project's functionality:
- Interactive image sharpening with custom mask drawing
- Batch processing of multiple images
- Model training control and monitoring
- Model export to ONNX format
- Before/after comparison views
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageTk
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, StringVar, IntVar, BooleanVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import time
from pathlib import Path
import glob
import webbrowser
import subprocess
from functools import partial

# Thêm lớp con của NavigationToolbar2Tk để tránh tự động pack()
class CustomNavigationToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent, pack_toolbar=False)

# Import project modules
from models.unet_attention import UNetAttention
from utils.data_utils import poisson_blend
from inference import load_image, load_mask, sharpen_image, visualize_results 

class AdvancedSharpeningApp:
    def __init__(self, model_path=None, device='cuda'):
        # Initialize parameters
        self.model_path = model_path
        self.device = device
        self.image_path = None
        self.image = None
        self.mask = None
        self.result = None
        self.original_result = None  # For comparison views
        self.drawing = False
        self.brush_size = 15
        self.eraser_mode = False
        self.last_x, self.last_y = None, None
        self.batch_pairs = []
        self.export_config = {
            "input_shape": "1,3,512,512",
            "optimize": True
        }
        
        # Try to load model (but don't block UI)
        self.model = None
        if model_path and os.path.exists(model_path):
            threading.Thread(target=self.load_model, daemon=True).start()
        
        # Setup UI
        self.setup_root_window()
        self.create_notebook_interface()
        self.setup_interactive_tab()
        self.setup_batch_tab()
        self.setup_export_tab()
        self.setup_settings_tab()
        self.setup_about_tab()
        
        # Status bar at the bottom
        self.setup_status_bar()
        
        # Set theme colors
        self.apply_theme()
        
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

    def setup_root_window(self):
        """Setup the main application window"""
        self.root = tk.Tk()
        self.root.title("Advanced Selective Image Sharpening")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 768)
        
        # Configure the grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Try to use a modern theme if available
        try:
            self.root.tk.call("source", "azure.tcl")
            self.root.tk.call("set_theme", "light")
        except:
            pass
            
    def create_notebook_interface(self):
        """Create tabbed interface"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create tabs
        self.interactive_tab = ttk.Frame(self.notebook)
        self.batch_tab = ttk.Frame(self.notebook)
        self.export_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.about_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.interactive_tab, text="Interactive Sharpening")
        self.notebook.add(self.batch_tab, text="Batch Processing")
        self.notebook.add(self.export_tab, text="Model Export")
        self.notebook.add(self.settings_tab, text="Settings")
        self.notebook.add(self.about_tab, text="About")
        
        # Configure grid for each tab
        for tab in [self.interactive_tab, self.batch_tab, self.export_tab, 
                    self.settings_tab, self.about_tab]:
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
    
    def setup_status_bar(self):
        """Setup status bar at the bottom of the window"""
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        
        # Status text
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Select a tab to begin.")
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, padx=5)
    
    def apply_theme(self):
        """Apply custom theme colors"""
        # Define colors
        self.bg_color = "#f5f5f5"
        self.accent_color = "#3498db"
        self.text_color = "#2c3e50"
        self.success_color = "#2ecc71"
        self.warning_color = "#f39c12"
        self.error_color = "#e74c3c"
        
        # Apply to widgets where possible
        style = ttk.Style()
        if 'vista' in style.theme_names():  # Use Vista theme as a base on Windows
            style.theme_use('vista')
        
        # Configure styles
        style.configure("TLabel", foreground=self.text_color)
        style.configure("TButton", foreground=self.text_color)
        style.configure("TFrame", background=self.bg_color)
        style.configure("TNotebook", background=self.bg_color)
        style.configure("Accent.TButton", background=self.accent_color, foreground="white")
        style.map("Accent.TButton",
                 background=[('active', self.accent_color), ('pressed', '#2980b9')],
                 foreground=[('active', 'white'), ('pressed', 'white')])
                 
        # Set background color for root window
        self.root.configure(background=self.bg_color)
        
        # ttk widgets don't support direct background configuration
        # (removed the loop to set background on tabs)
        
    def setup_interactive_tab(self):
        """Setup the interactive sharpening tab"""
        # Create main layout with panels
        panel_frame = ttk.Frame(self.interactive_tab)
        panel_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        panel_frame.columnconfigure(1, weight=1)
        panel_frame.rowconfigure(0, weight=1)
        
        # Left control panel
        left_panel = ttk.Frame(panel_frame, width=280)
        left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 5), pady=5)
        
        # Right image panel
        right_panel = ttk.Frame(panel_frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Setup left panel controls
        self.setup_interactive_controls(left_panel)
        
        # Setup right panel image display
        self.setup_interactive_display(right_panel)
        
    def setup_interactive_controls(self, parent):
        """Setup controls for the interactive tab"""
        # File operations section
        file_section = ttk.LabelFrame(parent, text="File Operations")
        file_section.pack(fill=tk.X, pady=5, padx=5, ipady=5)
        
        ttk.Button(file_section, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(file_section, text="Load Mask", command=self.load_mask_file).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(file_section, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(file_section, text="Save Mask", command=self.save_mask).pack(fill=tk.X, pady=2, padx=5)
        
        # Model selection
        model_section = ttk.LabelFrame(parent, text="Model")
        model_section.pack(fill=tk.X, pady=5, padx=5, ipady=5)
        
        self.model_path_var = tk.StringVar()
        self.model_path_var.set(self.model_path if self.model_path else "No model selected")
        
        ttk.Label(model_section, textvariable=self.model_path_var, 
                 wraplength=260).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(model_section, text="Select Model", 
                  command=self.select_model).pack(fill=tk.X, pady=2, padx=5)
        
        # Drawing tools section
        drawing_section = ttk.LabelFrame(parent, text="Drawing Tools")
        drawing_section.pack(fill=tk.X, pady=5, padx=5, ipady=5)
        
        # Tool mode radio buttons
        self.drawing_mode = tk.StringVar(value="brush")
        ttk.Radiobutton(drawing_section, text="Brush (Draw Mask)", variable=self.drawing_mode, 
                       value="brush", command=self.set_brush_mode).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(drawing_section, text="Eraser (Remove Mask)", variable=self.drawing_mode, 
                       value="eraser", command=self.set_eraser_mode).pack(anchor=tk.W, padx=5)
        
        # Brush size control
        ttk.Label(drawing_section, text=f"Brush Size: {self.brush_size}").pack(anchor=tk.W, pady=(10, 0), padx=5)
        
        size_frame = ttk.Frame(drawing_section)
        size_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(size_frame, text="1").pack(side=tk.LEFT)
        self.brush_slider = ttk.Scale(size_frame, from_=1, to=50, orient=tk.HORIZONTAL, 
                                   command=self.update_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(size_frame, text="50").pack(side=tk.LEFT)
        
        ttk.Button(drawing_section, text="Clear Mask", 
                  command=self.clear_mask).pack(fill=tk.X, pady=5, padx=5)
        
        # Processing section
        processing_section = ttk.LabelFrame(parent, text="Processing")
        processing_section.pack(fill=tk.X, pady=5, padx=5, ipady=5)
        
        # Processing options
        self.use_mask_channel = tk.BooleanVar(value=True)
        ttk.Checkbutton(processing_section, text="Use mask as input channel", 
                       variable=self.use_mask_channel).pack(anchor=tk.W, padx=5, pady=2)
                       
        self.use_poisson_blend = tk.BooleanVar(value=True)
        ttk.Checkbutton(processing_section, text="Use Poisson blending", 
                       variable=self.use_poisson_blend).pack(anchor=tk.W, padx=5, pady=2)
        
        # Processing button
        ttk.Button(processing_section, text="Process Image", 
                  command=self.process_image, style="Accent.TButton").pack(fill=tk.X, pady=5, padx=5)
        
        # View mode section
        view_section = ttk.LabelFrame(parent, text="View Mode")
        view_section.pack(fill=tk.X, pady=5, padx=5, ipady=5)
        
        self.view_mode = tk.StringVar(value="normal")
        ttk.Radiobutton(view_section, text="Normal View", variable=self.view_mode, 
                       value="normal", command=self.update_display).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(view_section, text="Before/After Split", variable=self.view_mode, 
                       value="split", command=self.update_display).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(view_section, text="Mask Overlay", variable=self.view_mode, 
                       value="overlay", command=self.update_display).pack(anchor=tk.W, padx=5)
    
    def setup_interactive_display(self, parent):
        """Setup the image display area for the interactive tab"""
        # Create figure for matplotlib
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.fig.subplots_adjust(wspace=0.05, hspace=0)
        
        # Initialize empty plots
        for ax in self.axes:
            ax.axis('off')
        
        self.axes[0].set_title("Original Image\nClick and drag to draw mask")
        self.axes[1].set_title("Mask")
        self.axes[2].set_title("Sharpened Result")
        
        # Create canvas for matplotlib figure
        self.canvas_frame = ttk.Frame(parent)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add navigation toolbar - Sử dụng lớp tùy chỉnh thay vì NavigationToolbar2Tk
        # Cần chỉ định pack_toolbar=False khi tạo toolbar để tránh gọi pack() tự động
        self.toolbar = CustomNavigationToolbar(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
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
            max_size = 1024
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                self.image = self.image.resize((new_width, new_height), Image.LANCZOS)
            
            # Create empty mask if one doesn't exist yet or sizes don't match
            if self.mask is None or self.mask.size != self.image.size:
                self.mask = Image.new('L', self.image.size, 0)
                self.mask_draw = ImageDraw.Draw(self.mask)
            
            # Clear previous result
            self.result = None
            self.original_result = None
            
            # Update display
            self.update_display()
            
            self.status_var.set(f"Loaded image: {os.path.basename(self.image_path)} ({self.image.width}x{self.image.height})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def load_mask_file(self):
        """Load a mask from file"""
        if self.image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
            
        mask_path = filedialog.askopenfilename(
            title="Select Mask Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not mask_path:
            return
            
        try:
            # Load mask and convert to grayscale
            loaded_mask = Image.open(mask_path).convert('L')
            
            # Resize to match current image
            loaded_mask = loaded_mask.resize(self.image.size, Image.LANCZOS)
            self.mask = loaded_mask
            self.mask_draw = ImageDraw.Draw(self.mask)
            
            # Update display
            self.update_display()
            
            self.status_var.set(f"Loaded mask: {os.path.basename(mask_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mask: {e}")
    
    def select_model(self):
        """Select a model checkpoint file"""
        model_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        
        if not model_path:
            return
            
        self.model_path = model_path
        self.model_path_var.set(os.path.basename(model_path))
        
        # Load the model in a separate thread
        threading.Thread(target=self.load_model, daemon=True).start()
        
    def set_brush_mode(self):
        """Set drawing mode to brush"""
        self.eraser_mode = False
        self.status_var.set("Brush mode: Drawing mask")
        
    def set_eraser_mode(self):
        """Set drawing mode to eraser"""
        self.eraser_mode = True
        self.status_var.set("Eraser mode: Removing mask")
        
    def update_brush_size(self, value):
        """Update the brush size"""
        self.brush_size = int(float(value))
        
    def clear_mask(self):
        """Clear the mask"""
        if self.image:
            self.mask = Image.new('L', self.image.size, 0)
            self.mask_draw = ImageDraw.Draw(self.mask)
            self.update_display()
            self.status_var.set("Mask cleared")
            
    def save_mask(self):
        """Save the current mask to a file"""
        if self.mask is None:
            messagebox.showinfo("Info", "No mask to save.")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Mask",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
            
        try:
            self.mask.save(save_path)
            self.status_var.set(f"Mask saved to {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save mask: {e}")
    
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
            self.status_var.set(f"Result saved to {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result: {e}")
            
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
        """Update the displayed images based on the current view mode"""
        # Clear axes
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
            
        # Display based on view mode
        if self.view_mode.get() == "normal":
            self._update_normal_view()
        elif self.view_mode.get() == "split":
            self._update_split_view()
        elif self.view_mode.get() == "overlay":
            self._update_overlay_view()
            
        # Update canvas
        self.canvas.draw()
            
    def _update_normal_view(self):
        """Update with normal three-panel view"""
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
    
    def _update_split_view(self):
        """Update with before/after split view"""
        if self.image is None:
            return
            
        # Display original in first panel
        self.axes[0].imshow(np.array(self.image))
        self.axes[0].set_title("Original Image")
        
        # Display mask in second panel
        if self.mask is not None:
            self.axes[1].imshow(np.array(self.mask), cmap='gray')
            self.axes[1].set_title("Mask")
        
        # Display split before/after in third panel
        if self.result is not None:
            # Create a split view
            h, w = self.result.shape[0], self.result.shape[1]
            split_point = w // 2
            
            # Combine original and result
            split_img = np.copy(self.result)
            orig_array = np.array(self.image) / 255.0
            split_img[:, :split_point, :] = orig_array[:, :split_point, :]
            
            # Display the split image
            self.axes[2].imshow(split_img)
            self.axes[2].set_title("Before | After")
            
            # Add a vertical line at the split point
            self.axes[2].axvline(x=split_point, color='white', linestyle='-', linewidth=2)
            
            # Add labels
            self.axes[2].text(split_point/4, 20, "Original", 
                             color='white', fontsize=12, ha='center',
                             bbox=dict(facecolor='black', alpha=0.5))
            self.axes[2].text(split_point + (w-split_point)/2, 20, "Sharpened", 
                             color='white', fontsize=12, ha='center',
                             bbox=dict(facecolor='black', alpha=0.5))
    
    def _update_overlay_view(self):
        """Update with mask overlay view"""
        if self.image is None:
            return
            
        # Display original in first panel
        self.axes[0].imshow(np.array(self.image))
        self.axes[0].set_title("Original Image")
        
        # Display mask in second panel
        if self.mask is not None:
            self.axes[1].imshow(np.array(self.mask), cmap='gray')
            self.axes[1].set_title("Mask")
        
        # Display original with mask overlay in third panel
        if self.image is not None and self.mask is not None:
            # Create a colored mask for overlay
            mask_array = np.array(self.mask)
            colored_mask = np.zeros((*mask_array.shape, 4), dtype=np.float32)
            colored_mask[:, :, 0] = 1.0  # Red
            colored_mask[:, :, 3] = mask_array / 255.0 * 0.5  # Alpha (50% of mask value)
            
            # Show the original image
            self.axes[2].imshow(np.array(self.image))
            
            # Overlay the colored mask
            self.axes[2].imshow(colored_mask)
            self.axes[2].set_title("Mask Overlay (Red)")
            
    def process_image(self):
        """Process the image using the model"""
        if not self.image:
            messagebox.showinfo("Info", "Please load an image first.")
            return
            
        if not self.model:
            if not self.model_path or not os.path.exists(self.model_path):
                messagebox.showinfo("Info", "Please select a valid model checkpoint first.")
                return
            messagebox.showinfo("Info", "Model is still loading. Please wait and try again.")
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
            # Get processing options
            use_mask_channel = self.use_mask_channel.get()
            use_poisson_blend = self.use_poisson_blend.get()
            
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
            input_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()
            output_np = output[0].cpu().permute(1, 2, 0).numpy()
            mask_np = mask_tensor[0].cpu().squeeze().numpy()
            
            if use_poisson_blend:
                self.root.after(0, lambda: self.status_var.set("Applying Poisson blending..."))
                # Apply Poisson blending
                self.result = poisson_blend(input_np, output_np, mask_np)
            else:
                # Simple alpha blending
                alpha = mask_np[:, :, np.newaxis]
                self.result = input_np * (1 - alpha) + output_np * alpha
            
            # Store the original result for comparison
            self.original_result = output_np
            
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

    def setup_batch_tab(self):
        """Setup the batch processing tab"""
        # Create main layout
        main_frame = ttk.Frame(self.batch_tab)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Directory selection
        dir_frame = ttk.LabelFrame(control_frame, text="Directories")
        dir_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Input directory
        input_frame = ttk.Frame(dir_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(input_frame, text="Input Directory:").pack(side=tk.LEFT)
        self.input_dir_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_dir_var, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Browse", command=self.select_input_dir).pack(side=tk.LEFT)
        
        # Mask directory
        mask_frame = ttk.Frame(dir_frame)
        mask_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(mask_frame, text="Mask Directory:").pack(side=tk.LEFT)
        self.mask_dir_var = tk.StringVar()
        ttk.Entry(mask_frame, textvariable=self.mask_dir_var, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(mask_frame, text="Browse", command=self.select_mask_dir).pack(side=tk.LEFT)
        
        # Output directory
        output_frame = ttk.Frame(dir_frame)
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar(value="results")
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.select_output_dir).pack(side=tk.LEFT)
        
        # Options frame
        options_frame = ttk.LabelFrame(control_frame, text="Processing Options")
        options_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Model selection
        model_frame = ttk.Frame(options_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.batch_model_path_var = tk.StringVar()
        self.batch_model_path_var.set(self.model_path if self.model_path else "No model selected")
        ttk.Entry(model_frame, textvariable=self.batch_model_path_var, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="Browse", command=self.select_batch_model).pack(side=tk.LEFT)
        
        # Processing options
        options_inner_frame = ttk.Frame(options_frame)
        options_inner_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Left column
        left_options = ttk.Frame(options_inner_frame)
        left_options.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.batch_use_mask_channel = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_options, text="Use mask as input channel", 
                       variable=self.batch_use_mask_channel).pack(anchor=tk.W, pady=2)
                       
        self.batch_use_poisson_blend = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_options, text="Use Poisson blending", 
                       variable=self.batch_use_poisson_blend).pack(anchor=tk.W, pady=2)
        
        # Right column
        right_options = ttk.Frame(options_inner_frame)
        right_options.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.batch_match_filenames = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_options, text="Match files by name", 
                       variable=self.batch_match_filenames).pack(anchor=tk.W, pady=2)
                       
        self.batch_save_visualizations = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_options, text="Save visualizations", 
                       variable=self.batch_save_visualizations).pack(anchor=tk.W, pady=2)
        
        # Action buttons
        actions_frame = ttk.Frame(control_frame)
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(actions_frame, text="Scan Directories", 
                  command=self.scan_batch_directories).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(actions_frame, text="Process All", 
                  command=self.process_batch, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(actions_frame, text="Stop", 
                  command=self.stop_batch_processing).pack(side=tk.LEFT, padx=5)
        
        # Results list
        list_frame = ttk.LabelFrame(main_frame, text="Files to Process")
        list_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Create scrollable list
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        columns = ('input_file', 'mask_file', 'status')
        self.batch_list = ttk.Treeview(list_frame, columns=columns, show='headings', 
                                      yscrollcommand=scrollbar.set)
        
        self.batch_list.heading('input_file', text='Input File')
        self.batch_list.heading('mask_file', text='Mask File')
        self.batch_list.heading('status', text='Status')
        
        self.batch_list.column('input_file', width=300)
        self.batch_list.column('mask_file', width=300)
        self.batch_list.column('status', width=100)
        
        self.batch_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.batch_list.yview)
        
        # Batch processing variables
        self.batch_processing = False
        self.batch_pairs = []
        self.batch_results = []
    
    def select_input_dir(self):
        """Select input directory for batch processing"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir_var.set(directory)
            self.scan_batch_directories()
    
    def select_mask_dir(self):
        """Select mask directory for batch processing"""
        directory = filedialog.askdirectory(title="Select Mask Directory")
        if directory:
            self.mask_dir_var.set(directory)
            self.scan_batch_directories()
    
    def select_output_dir(self):
        """Select output directory for batch processing"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def select_batch_model(self):
        """Select model for batch processing"""
        model_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        
        if model_path:
            self.batch_model_path_var.set(model_path)
            # Update main model path too
            self.model_path = model_path
            if hasattr(self, 'model_path_var'):
                self.model_path_var.set(os.path.basename(model_path))
            
            # Load the model if not loaded yet
            if self.model is None:
                threading.Thread(target=self.load_model, daemon=True).start()
    
    def scan_batch_directories(self):
        """Scan input and mask directories and build file pairs"""
        input_dir = self.input_dir_var.get()
        mask_dir = self.mask_dir_var.get()
        
        if not input_dir or not os.path.exists(input_dir):
            messagebox.showinfo("Info", "Please select a valid input directory.")
            return
            
        if not mask_dir or not os.path.exists(mask_dir):
            messagebox.showinfo("Info", "Please select a valid mask directory.")
            return
        
        try:
            self.status_var.set(f"Scanning directories...")
            
            # Clear existing items
            for item in self.batch_list.get_children():
                self.batch_list.delete(item)
            
            # Find image files in input directory
            input_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
                                glob.glob(os.path.join(input_dir, '*.jpeg')) + 
                                glob.glob(os.path.join(input_dir, '*.png')) + 
                                glob.glob(os.path.join(input_dir, '*.bmp')))
            
            # Find mask files in mask directory
            mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.jpg')) + 
                               glob.glob(os.path.join(mask_dir, '*.jpeg')) + 
                               glob.glob(os.path.join(mask_dir, '*.png')) + 
                               glob.glob(os.path.join(mask_dir, '*.bmp')))
            
            if not input_files:
                self.status_var.set(f"No image files found in {input_dir}")
                return
                
            if not mask_files:
                self.status_var.set(f"No mask files found in {mask_dir}")
                return
            
            # Match files based on option
            if self.batch_match_filenames.get():
                # Match by filename
                input_basenames = {Path(f).stem: f for f in input_files}
                mask_basenames = {Path(f).stem: f for f in mask_files}
                
                # Find common names
                common_names = set(input_basenames.keys()) & set(mask_basenames.keys())
                
                if not common_names:
                    messagebox.showinfo("Info", "No matching filenames found. Falling back to index-based matching.")
                    self.batch_pairs = [(input_files[i], mask_files[i % len(mask_files)]) for i in range(len(input_files))]
                else:
                    self.batch_pairs = [(input_basenames[name], mask_basenames[name]) for name in sorted(common_names)]
            else:
                # Match by index (cycling mask files if needed)
                self.batch_pairs = [(input_files[i], mask_files[i % len(mask_files)]) for i in range(len(input_files))]
            
            # Populate the treeview
            for i, (input_path, mask_path) in enumerate(self.batch_pairs):
                self.batch_list.insert('', 'end', values=(
                    os.path.basename(input_path),
                    os.path.basename(mask_path),
                    'Pending'
                ))
            
            self.status_var.set(f"Found {len(self.batch_pairs)} image pairs to process")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan directories: {e}")
            self.status_var.set(f"Error scanning directories: {e}")
    
    def process_batch(self):
        """Process all images in the batch"""
        # Check if we have pairs to process
        if not self.batch_pairs:
            messagebox.showinfo("Info", "No files to process. Please scan directories first.")
            return
            
        # Check for model
        model_path = self.batch_model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showinfo("Info", "Please select a valid model checkpoint.")
            return
            
        # Create output directory
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showinfo("Info", "Please specify an output directory.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Start processing in a separate thread
        self.batch_processing = True
        threading.Thread(target=self._process_batch_thread).start()
    
    def stop_batch_processing(self):
        """Stop the current batch processing"""
        if self.batch_processing:
            self.batch_processing = False
            self.status_var.set("Batch processing stopped by user")
    
    def _process_batch_thread(self):
        """Thread function to process all images in batch"""
        self.batch_results = []
        start_time = time.time()
        
        # Get options
        model_path = self.batch_model_path_var.get()
        output_dir = self.output_dir_var.get()
        use_mask_channel = self.batch_use_mask_channel.get()
        use_poisson_blend = self.batch_use_poisson_blend.get()
        save_visualizations = self.batch_save_visualizations.get()
        
        # Load model if needed
        if self.model is None:
            self.root.after(0, lambda: self.status_var.set("Loading model..."))
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model = UNetAttention(n_channels=3, n_classes=3, with_mask_channel=use_mask_channel)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error loading model: {e}"))
                self.batch_processing = False
                return
        
        # Process each pair
        total = len(self.batch_pairs)
        for i, (input_path, mask_path) in enumerate(self.batch_pairs):
            if not self.batch_processing:
                break
                
            # Update status
            self.root.after(0, lambda: self.status_var.set(f"Processing {i+1}/{total}: {os.path.basename(input_path)}"))
            self.root.after(0, lambda: self.progress_var.set((i / total) * 100))
            
            # Update list status
            item_id = self.batch_list.get_children()[i]
            self.root.after(0, lambda id=item_id: self.batch_list.item(id, values=(
                self.batch_list.item(id)['values'][0],
                self.batch_list.item(id)['values'][1],
                'Processing'
            )))
            
            try:
                # Load input image
                input_image = load_image(input_path)
                
                # Get image dimensions for mask resizing
                input_size = (input_image.shape[3], input_image.shape[2])  # width, height
                
                # Load and resize mask
                mask = load_mask(mask_path, size=input_size)
                
                # Create args object for sharpen_image
                args = argparse.Namespace(
                    device=self.device,
                    use_mask_channel=use_mask_channel,
                    use_poisson_blend=use_poisson_blend
                )
                
                # Apply sharpening
                sharpened, result = sharpen_image(self.model, input_image, mask, args)
                
                # Create output filename
                input_filename = os.path.basename(input_path)
                output_base = os.path.splitext(input_filename)[0]
                
                # Save result image
                result_path = os.path.join(output_dir, f"{output_base}_sharpened.png")
                img_result = Image.fromarray((result[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                img_result.save(result_path)
                
                # Save visualization if requested
                if save_visualizations:
                    viz_path = os.path.join(output_dir, f"{output_base}_visualization.png")
                    visualize_results(input_image, sharpened, result, mask, viz_path)
                
                # Add to results
                self.batch_results.append((input_path, result_path))
                
                # Update list status
                self.root.after(0, lambda id=item_id: self.batch_list.item(id, values=(
                    self.batch_list.item(id)['values'][0],
                    self.batch_list.item(id)['values'][1],
                    'Completed'
                )))
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                # Update list status
                self.root.after(0, lambda id=item_id: self.batch_list.item(id, values=(
                    self.batch_list.item(id)['values'][0],
                    self.batch_list.item(id)['values'][1],
                    f'Error: {str(e)[:20]}'
                )))
        
        elapsed_time = time.time() - start_time
        
        # Update final status
        if self.batch_processing:
            self.root.after(0, lambda: self.status_var.set(
                f"Batch processing complete! Processed {len(self.batch_results)}/{total} "
                f"images in {elapsed_time:.2f} seconds ({elapsed_time/max(1, len(self.batch_results)):.2f}s per image)"
            ))
        
        self.batch_processing = False
        self.root.after(0, lambda: self.progress_var.set(100))
        self.root.after(1000, lambda: self.progress_var.set(0)) 

    def setup_export_tab(self):
        """Setup the model export tab"""
        # Create main layout
        main_frame = ttk.Frame(self.export_tab)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Model export frame
        export_frame = ttk.LabelFrame(main_frame, text="Export Model to ONNX Format")
        export_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Model selection
        model_frame = ttk.Frame(export_frame)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_model_path_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.export_model_path_var, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(model_frame, text="Browse", command=self.select_export_model).grid(row=0, column=2, padx=5, pady=5)
        
        # Output file
        ttk.Label(model_frame, text="Output Path:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_output_path_var = tk.StringVar(value="model.onnx")
        ttk.Entry(model_frame, textvariable=self.export_output_path_var, width=50).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(model_frame, text="Browse", command=self.select_export_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Export options
        options_frame = ttk.LabelFrame(export_frame, text="Export Options")
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Input shape
        shape_frame = ttk.Frame(options_frame)
        shape_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(shape_frame, text="Input Shape:").pack(side=tk.LEFT, padx=5)
        self.export_input_shape_var = tk.StringVar(value="1,3,512,512")
        ttk.Entry(shape_frame, textvariable=self.export_input_shape_var, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(shape_frame, text="(batch_size,channels,height,width)").pack(side=tk.LEFT, padx=5)
        
        # Optimization options
        optimize_frame = ttk.Frame(options_frame)
        optimize_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.export_optimize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(optimize_frame, text="Apply ONNX optimizations for faster inference", 
                      variable=self.export_optimize_var).pack(anchor=tk.W)
        
        self.export_use_mask_channel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(optimize_frame, text="Model uses mask as input channel", 
                      variable=self.export_use_mask_channel_var).pack(anchor=tk.W)
        
        # Test with sample image
        test_frame = ttk.LabelFrame(export_frame, text="Test Exported Model")
        test_frame.pack(fill=tk.X, padx=10, pady=10)
        
        test_inner_frame = ttk.Frame(test_frame)
        test_inner_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.export_test_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(test_inner_frame, text="Test export with sample images", 
                      variable=self.export_test_var, command=self.toggle_test_export).pack(anchor=tk.W)
        
        # Sample image selection (initially disabled)
        self.test_image_frame = ttk.Frame(test_frame)
        self.test_image_frame.pack(fill=tk.X, padx=10, pady=10)
        self.test_image_frame.pack_forget()  # Hide initially
        
        ttk.Label(self.test_image_frame, text="Test Image:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_test_image_var = tk.StringVar()
        ttk.Entry(self.test_image_frame, textvariable=self.export_test_image_var, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(self.test_image_frame, text="Browse", command=self.select_test_image).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(self.test_image_frame, text="Test Mask:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_test_mask_var = tk.StringVar()
        ttk.Entry(self.test_image_frame, textvariable=self.export_test_mask_var, width=50).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(self.test_image_frame, text="Browse", command=self.select_test_mask).grid(row=1, column=2, padx=5, pady=5)
        
        # Export button
        export_button_frame = ttk.Frame(export_frame)
        export_button_frame.pack(padx=10, pady=20)
        
        ttk.Button(export_button_frame, text="Export Model", 
                  command=self.export_model, style="Accent.TButton").pack(padx=10, ipadx=20, ipady=10)
        
        # Information area
        info_frame = ttk.LabelFrame(main_frame, text="Export Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.export_info_text = tk.Text(info_frame, wrap=tk.WORD, height=10)
        self.export_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial info text
        export_info = """ONNX Export Information:

1. Exporting to ONNX format allows faster inference in production environments.
2. The exported model can be used with ONNX Runtime on various platforms.
3. Optimizations can significantly improve inference speed.
4. Test your exported model with sample inputs to ensure it works correctly.

Example Python code for inference with ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Load and preprocess input image
img = Image.open("input.jpg").convert('RGB')
transform = transforms.ToTensor()
input_tensor = transform(img).unsqueeze(0).numpy()

# Load and preprocess mask
mask = Image.open("mask.png").convert('L')
mask_tensor = transform(mask).unsqueeze(0).numpy()

# Run inference
input_name = session.get_inputs()[0].name
mask_name = session.get_inputs()[1].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: input_tensor, mask_name: mask_tensor})[0]

# Post-process and save result
result_array = np.clip(result[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
result_img = Image.fromarray(result_array)
result_img.save("result.png")
```
"""
        self.export_info_text.insert(tk.END, export_info)
        self.export_info_text.config(state=tk.DISABLED)
    
    def select_export_model(self):
        """Select model for export"""
        model_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        
        if model_path:
            self.export_model_path_var.set(model_path)
            # Update other model path variables
            self.model_path = model_path
            if hasattr(self, 'model_path_var'):
                self.model_path_var.set(os.path.basename(model_path))
            if hasattr(self, 'batch_model_path_var'):
                self.batch_model_path_var.set(model_path)
    
    def select_export_output(self):
        """Select output path for the exported model"""
        output_path = filedialog.asksaveasfilename(
            title="Save ONNX Model As",
            defaultextension=".onnx",
            filetypes=[("ONNX Models", "*.onnx"), ("All Files", "*.*")]
        )
        
        if output_path:
            self.export_output_path_var.set(output_path)
    
    def toggle_test_export(self):
        """Show/hide test image fields based on checkbox"""
        if self.export_test_var.get():
            self.test_image_frame.pack(fill=tk.X, padx=10, pady=10)
        else:
            self.test_image_frame.pack_forget()
    
    def select_test_image(self):
        """Select test image for export testing"""
        image_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if image_path:
            self.export_test_image_var.set(image_path)
    
    def select_test_mask(self):
        """Select test mask for export testing"""
        mask_path = filedialog.askopenfilename(
            title="Select Test Mask",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if mask_path:
            self.export_test_mask_var.set(mask_path)
    
    def export_model(self):
        """Export the model to ONNX format"""
        # Check if we have required parameters
        model_path = self.export_model_path_var.get()
        output_path = self.export_output_path_var.get()
        
        if not model_path or not os.path.exists(model_path):
            messagebox.showinfo("Info", "Please select a valid model checkpoint.")
            return
            
        if not output_path:
            messagebox.showinfo("Info", "Please specify an output path for the ONNX model.")
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get export options
        input_shape = self.export_input_shape_var.get()
        optimize = self.export_optimize_var.get()
        use_mask_channel = self.export_use_mask_channel_var.get()
        test_export = self.export_test_var.get()
        
        # Get test image paths if testing
        test_image = self.export_test_image_var.get() if test_export else None
        test_mask = self.export_test_mask_var.get() if test_export else None
        
        if test_export and (not test_image or not os.path.exists(test_image) or 
                          not test_mask or not os.path.exists(test_mask)):
            messagebox.showinfo("Info", "Please select valid test image and mask files.")
            return
        
        # Start export process in a separate thread
        threading.Thread(target=lambda: self._export_model_thread(
            model_path, output_path, input_shape, optimize, use_mask_channel, test_image, test_mask
        )).start()
    
    def _export_model_thread(self, model_path, output_path, input_shape, optimize, use_mask_channel, test_image, test_mask):
        """Thread function to export the model"""
        try:
            self.root.after(0, lambda: self.status_var.set("Exporting model to ONNX format..."))
            self.root.after(0, lambda: self.progress_var.set(10))
            
            # Load the model
            checkpoint = torch.load(model_path, map_location='cpu')
            model = UNetAttention(n_channels=3, n_classes=3, with_mask_channel=use_mask_channel)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.root.after(0, lambda: self.progress_var.set(30))
            
            # Parse input shape
            shape_parts = input_shape.split(',')
            if len(shape_parts) != 4:
                raise ValueError("Input shape must be in format: batch_size,channels,height,width")
                
            input_shape_tuple = tuple(map(int, shape_parts))
            mask_shape_tuple = (input_shape_tuple[0], 1, input_shape_tuple[2], input_shape_tuple[3])
            
            # Export to ONNX
            self.root.after(0, lambda: self.status_var.set(f"Exporting model with input shape {input_shape}..."))
            
            dummy_input = torch.randn(input_shape_tuple)
            dummy_mask = torch.randn(mask_shape_tuple)
            
            torch.onnx.export(
                model,
                (dummy_input, dummy_mask),
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input', 'mask'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'mask': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                }
            )
            
            self.root.after(0, lambda: self.progress_var.set(60))
            
            # Optimize if requested
            if optimize:
                self.root.after(0, lambda: self.status_var.set("Applying ONNX optimizations..."))
                try:
                    import onnx
                    import onnxoptimizer
                    
                    onnx_model = onnx.load(output_path)
                    passes = [
                        "eliminate_unused_initializer",
                        "eliminate_deadend",
                        "eliminate_identity",
                        "fuse_add_bias_into_conv",
                        "fuse_bn_into_conv",
                        "fuse_consecutive_squeezes",
                        "fuse_consecutive_transposes",
                        "fuse_matmul_add_bias_into_gemm",
                        "fuse_pad_into_conv",
                        "fuse_transpose_into_gemm"
                    ]
                    
                    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
                    onnx.save(optimized_model, output_path)
                    self.root.after(0, lambda: self.status_var.set("ONNX optimizations applied successfully"))
                except ImportError:
                    self.root.after(0, lambda: self.status_var.set(
                        "ONNX optimizer not found. Install with: pip install onnxoptimizer"
                    ))
            
            self.root.after(0, lambda: self.progress_var.set(80))
            
            # Test if requested
            if test_image and test_mask:
                self.root.after(0, lambda: self.status_var.set("Testing exported model with sample images..."))
                try:
                    import onnxruntime as ort
                    
                    # Load and process test image
                    transform = transforms.ToTensor()
                    img = Image.open(test_image).convert('RGB')
                    input_tensor = transform(img).unsqueeze(0).numpy()
                    
                    # Load and process test mask
                    mask = Image.open(test_mask).convert('L')
                    mask_tensor = transform(mask).unsqueeze(0).numpy()
                    
                    # Run inference with ONNX Runtime
                    session = ort.InferenceSession(output_path)
                    input_name = session.get_inputs()[0].name
                    mask_name = session.get_inputs()[1].name
                    output_name = session.get_outputs()[0].name
                    
                    result = session.run([output_name], {input_name: input_tensor, mask_name: mask_tensor})[0]
                    
                    # Save test result
                    output_dir = os.path.dirname(output_path)
                    result_path = os.path.join(output_dir, 'onnx_test_result.png')
                    
                    result_array = np.clip(result[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                    result_img = Image.fromarray(result_array)
                    result_img.save(result_path)
                    
                    self.root.after(0, lambda: self.status_var.set(
                        f"Model tested successfully. Result saved to {result_path}"
                    ))
                except ImportError:
                    self.root.after(0, lambda: self.status_var.set(
                        "ONNX Runtime not found. Install with: pip install onnxruntime"
                    ))
                except Exception as e:
                    self.root.after(0, lambda: self.status_var.set(f"Error testing model: {e}"))
            
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.status_var.set(f"Model exported successfully to {output_path}"))
            self.root.after(1000, lambda: self.progress_var.set(0))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error exporting model: {e}"))
            self.root.after(0, lambda: self.progress_var.set(0))
            messagebox.showerror("Error", f"Failed to export model: {e}")

    def setup_settings_tab(self):
        """Setup the settings tab"""
        main_frame = ttk.Frame(self.settings_tab)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # General settings
        general_frame = ttk.LabelFrame(main_frame, text="General Settings")
        general_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # GPU/CPU option
        device_frame = ttk.Frame(general_frame)
        device_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT, padx=5)
        self.device_var = tk.StringVar(value=self.device)
        device_combobox = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                       values=["cuda", "cpu"], width=10, state="readonly")
        device_combobox.pack(side=tk.LEFT, padx=5)
        
        # Update button
        ttk.Button(device_frame, text="Update", 
                  command=self.update_device_setting).pack(side=tk.LEFT, padx=5)
        
        # Show device info
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_info = f"CUDA is available: {device_name}"
        else:
            device_info = "CUDA is not available, using CPU"
        
        ttk.Label(general_frame, text=device_info).pack(anchor=tk.W, padx=10, pady=5)
        
        # UI settings
        ui_frame = ttk.LabelFrame(main_frame, text="User Interface Settings")
        ui_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # UI scale
        scale_frame = ttk.Frame(ui_frame)
        scale_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(scale_frame, text="UI Scale:").pack(side=tk.LEFT, padx=5)
        
        self.ui_scale_var = tk.DoubleVar(value=1.0)
        scale_slider = ttk.Scale(scale_frame, from_=0.8, to=1.5, variable=self.ui_scale_var, 
                                orient=tk.HORIZONTAL, length=200)
        scale_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Label(scale_frame, textvariable=tk.StringVar(value=lambda: f"{self.ui_scale_var.get():.1f}")).pack(side=tk.LEFT, padx=5)
        
        # Apply scale button
        ttk.Button(scale_frame, text="Apply", 
                  command=self.apply_ui_scale).pack(side=tk.LEFT, padx=5)
        
        # Theme selection
        theme_frame = ttk.Frame(ui_frame)
        theme_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(theme_frame, text="Color Theme:").pack(side=tk.LEFT, padx=5)
        
        self.theme_var = tk.StringVar(value="Light")
        theme_combobox = ttk.Combobox(theme_frame, textvariable=self.theme_var, 
                                     values=["Light", "Dark"], width=10, state="readonly")
        theme_combobox.pack(side=tk.LEFT, padx=5)
        
        # Apply theme button
        ttk.Button(theme_frame, text="Apply Theme", 
                  command=self.apply_theme_setting).pack(side=tk.LEFT, padx=5)
        
        # Path settings
        paths_frame = ttk.LabelFrame(main_frame, text="Default Paths")
        paths_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Default model path
        model_frame = ttk.Frame(paths_frame)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="Default Model:").pack(side=tk.LEFT, padx=5)
        self.default_model_var = tk.StringVar(value=self.model_path if self.model_path else "")
        ttk.Entry(model_frame, textvariable=self.default_model_var, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="Browse", 
                  command=self.set_default_model).pack(side=tk.LEFT, padx=5)
        
        # Default output directory
        output_frame = ttk.Frame(paths_frame)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        self.default_output_var = tk.StringVar(value="results")
        ttk.Entry(output_frame, textvariable=self.default_output_var, width=40).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", 
                  command=self.set_default_output).pack(side=tk.LEFT, padx=5)
        
        # Save settings button
        save_frame = ttk.Frame(main_frame)
        save_frame.pack(pady=20)
        
        ttk.Button(save_frame, text="Save Settings", 
                  command=self.save_settings, style="Accent.TButton").pack(ipadx=20, ipady=5)
    
    def setup_about_tab(self):
        """Setup the about tab"""
        main_frame = ttk.Frame(self.about_tab)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # App title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(pady=20)
        
        ttk.Label(title_frame, text="Selective Image Sharpening", 
                 font=("Arial", 16, "bold")).pack()
        
        ttk.Label(title_frame, text="Advanced User Interface", 
                 font=("Arial", 12)).pack()
        
        ttk.Label(title_frame, text="Version 1.0").pack()
        
        # Description
        desc_frame = ttk.LabelFrame(main_frame, text="About this Project")
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        description = """This application implements an AI agent for selective image sharpening, which enhances the sharpness of specific regions in an image while keeping the rest unchanged.

Key Features:
• Selectively sharpen regions of an image defined by a mask
• U-Net architecture with attention gates for focused enhancement 
• Multiple loss functions for natural results
• Poisson blending for seamless integration of sharpened regions
• Interactive interface with intuitive mask drawing tools
• Batch processing capabilities for multiple images
• Model export to ONNX for fast production deployment

The application uses deep learning techniques to analyze and enhance images, focusing specifically on the regions you select with the mask tool."""
        
        desc_text = tk.Text(desc_frame, wrap=tk.WORD, height=10)
        desc_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        desc_text.insert(tk.END, description)
        desc_text.config(state=tk.DISABLED)
        
        # Links and buttons
        links_frame = ttk.Frame(main_frame)
        links_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(links_frame, text="View Project on GitHub", 
                  command=lambda: webbrowser.open("https://github.com/yourusername/selective-sharpening")).pack(pady=5)
        
        ttk.Button(links_frame, text="View Documentation", 
                  command=lambda: webbrowser.open("https://github.com/yourusername/selective-sharpening/blob/main/README.md")).pack(pady=5)
        
        # Credits
        credits_frame = ttk.LabelFrame(main_frame, text="Credits")
        credits_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(credits_frame, text="Developed by: Your Name", 
                 font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=2)
        
        ttk.Label(credits_frame, text="Built with: PyTorch, Tkinter, OpenCV", 
                 font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=2)
        
        ttk.Label(credits_frame, text="License: MIT", 
                 font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=2)
    
    def update_device_setting(self):
        """Update the device setting (CUDA or CPU)"""
        device = self.device_var.get()
        if device == "cuda" and not torch.cuda.is_available():
            messagebox.showinfo("Info", "CUDA is not available on this system. Using CPU instead.")
            self.device_var.set("cpu")
            device = "cpu"
        
        self.device = device
        
        # If model is loaded, move it to the new device
        if self.model is not None:
            try:
                self.model.to(self.device)
                self.status_var.set(f"Model moved to {self.device}")
            except Exception as e:
                self.status_var.set(f"Error moving model to {self.device}: {e}")
    
    def apply_ui_scale(self):
        """Apply UI scale setting"""
        scale = self.ui_scale_var.get()
        # Set font scaling if possible
        try:
            default_font = ("TkDefaultFont", int(10 * scale))
            self.root.option_add("*Font", default_font)
            self.status_var.set(f"UI scale set to {scale:.1f}")
        except Exception as e:
            self.status_var.set(f"Error applying UI scale: {e}")
    
    def apply_theme_setting(self):
        """Apply theme setting"""
        theme = self.theme_var.get()
        if theme == "Dark":
            # Try to set dark theme
            try:
                self.root.tk.call("set_theme", "dark")
                self.status_var.set("Dark theme applied")
            except:
                self.status_var.set("Failed to apply dark theme")
        else:
            # Try to set light theme
            try:
                self.root.tk.call("set_theme", "light")
                self.status_var.set("Light theme applied")
            except:
                self.status_var.set("Failed to apply light theme")
    
    def set_default_model(self):
        """Set default model path"""
        model_path = filedialog.askopenfilename(
            title="Select Default Model",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        
        if model_path:
            self.default_model_var.set(model_path)
    
    def set_default_output(self):
        """Set default output directory"""
        directory = filedialog.askdirectory(title="Select Default Output Directory")
        if directory:
            self.default_output_var.set(directory)
    
    def save_settings(self):
        """Save settings to configuration file"""
        try:
            # Create settings directory if it doesn't exist
            os.makedirs('settings', exist_ok=True)
            
            # Prepare settings dictionary
            settings = {
                'device': self.device_var.get(),
                'ui_scale': self.ui_scale_var.get(),
                'theme': self.theme_var.get(),
                'default_model': self.default_model_var.get(),
                'default_output': self.default_output_var.get()
            }
            
            # Save to file using JSON
            import json
            with open('settings/config.json', 'w') as f:
                json.dump(settings, f, indent=4)
                
            self.status_var.set("Settings saved successfully")
        except Exception as e:
            self.status_var.set(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def load_settings(self):
        """Load settings from configuration file"""
        try:
            import json
            if os.path.exists('settings/config.json'):
                with open('settings/config.json', 'r') as f:
                    settings = json.load(f)
                    
                # Apply settings
                if 'device' in settings:
                    self.device = settings['device']
                    if hasattr(self, 'device_var'):
                        self.device_var.set(settings['device'])
                
                if 'ui_scale' in settings:
                    if hasattr(self, 'ui_scale_var'):
                        self.ui_scale_var.set(settings['ui_scale'])
                
                if 'theme' in settings:
                    if hasattr(self, 'theme_var'):
                        self.theme_var.set(settings['theme'])
                
                if 'default_model' in settings and settings['default_model']:
                    self.model_path = settings['default_model']
                    if hasattr(self, 'default_model_var'):
                        self.default_model_var.set(settings['default_model'])
                    if hasattr(self, 'model_path_var'):
                        self.model_path_var.set(os.path.basename(settings['default_model']))
                    if hasattr(self, 'batch_model_path_var'):
                        self.batch_model_path_var.set(settings['default_model'])
                    if hasattr(self, 'export_model_path_var'):
                        self.export_model_path_var.set(settings['default_model'])
                
                if 'default_output' in settings:
                    if hasattr(self, 'output_dir_var'):
                        self.output_dir_var.set(settings['default_output'])
                    if hasattr(self, 'default_output_var'):
                        self.default_output_var.set(settings['default_output'])
                
                self.status_var.set("Settings loaded successfully")
        except Exception as e:
            self.status_var.set(f"Error loading settings: {e}")
    
    def run(self):
        """Run the application"""
        # Load settings
        self.load_settings()
        
        # Start the main loop
        self.root.mainloop()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Selective Image Sharpening UI')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Start the application
    app = AdvancedSharpeningApp(model_path=args.model_path, device=args.device)
    app.run()

if __name__ == '__main__':
    main() 