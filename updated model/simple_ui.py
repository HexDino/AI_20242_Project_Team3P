#!/usr/bin/env python
"""
Simple User Interface for Selective Image Sharpening

Giao diện đơn giản và thân thiện với người dùng cho việc làm nét chọn lọc hình ảnh
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# Import project modules
from models.unet_attention import UNetAttention
from utils.data_utils import poisson_blend

class SimpleImageSharpener:
    def __init__(self, model_path=None, device='cuda'):
        # Initialize parameters
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.image_path = None
        self.image = None
        self.mask = None
        self.result = None
        self.brush_size = 15
        self.eraser_mode = False
        self.drawing = False
        self.last_x, self.last_y = None, None
        
        # Setup UI
        self.setup_ui()
        
        # Try to load model (in background)
        self.model = None
        if model_path and os.path.exists(model_path):
            threading.Thread(target=self.load_model, daemon=True).start()
            
    def load_model(self):
        """Load the model in a separate thread to avoid blocking the UI"""
        try:
            self.status_var.set("Đang tải mô hình...")
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Initialize model
            self.model = UNetAttention(n_channels=3, n_classes=3, with_mask_channel=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Update UI
            self.status_var.set("Đã tải mô hình thành công!")
        except Exception as e:
            self.status_var.set(f"Lỗi tải mô hình: {e}")
            self.model = None
            
    def setup_ui(self):
        """Setup simple UI with focus on user friendliness"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Làm nét hình ảnh chọn lọc")
        self.root.geometry("1200x720")
        self.root.configure(bg="#f0f0f0")
        self.root.minsize(1000, 700)
        
        # Create main layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        
        # Right panel for image display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
        
        # Setup header with instructions
        header_frame = ttk.Frame(left_panel)
        header_frame.pack(fill=tk.X, pady=5)
        
        header_label = ttk.Label(header_frame, text="Làm nét vùng chọn trong ảnh", 
                               font=("Arial", 12, "bold"), wraplength=240)
        header_label.pack(pady=5)
        
        instruction_text = (
            "1. Tải ảnh lên\n"
            "2. Vẽ vùng cần làm nét\n"
            "3. Nhấn 'Xử lý ảnh'\n"
            "4. Lưu kết quả"
        )
        
        instruction_label = ttk.Label(header_frame, text=instruction_text, 
                                    wraplength=240, justify=tk.LEFT)
        instruction_label.pack(pady=5)
        
        # Setup image controls
        image_section = ttk.LabelFrame(left_panel, text="Thao tác với ảnh")
        image_section.pack(fill=tk.X, pady=10)
        
        ttk.Button(image_section, text="Tải ảnh lên", 
                  command=self.load_image).pack(fill=tk.X, pady=5, padx=5)
        
        # Model selection
        model_frame = ttk.Frame(image_section)
        model_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.model_label = ttk.Label(model_frame, text="Mô hình: " + 
                                   (os.path.basename(self.model_path) if self.model_path else "Chưa chọn"))
        self.model_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(model_frame, text="Chọn", 
                  command=self.select_model, width=8).pack(side=tk.RIGHT)
        
        # Drawing tools section
        drawing_section = ttk.LabelFrame(left_panel, text="Công cụ vẽ")
        drawing_section.pack(fill=tk.X, pady=10)
        
        # Tool mode frame
        tool_frame = ttk.Frame(drawing_section)
        tool_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.tool_var = tk.StringVar(value="brush")
        
        def set_brush():
            self.tool_var.set("brush")
            self.eraser_mode = False
            self.status_var.set("Chế độ bút: Vẽ vùng cần làm nét")
            
        def set_eraser():
            self.tool_var.set("eraser")
            self.eraser_mode = True
            self.status_var.set("Chế độ tẩy: Xóa phần vùng đã vẽ")
        
        # Tool buttons frame
        tool_buttons = ttk.Frame(tool_frame)
        tool_buttons.pack(fill=tk.X)
        
        brush_btn = ttk.Button(tool_buttons, text="Bút vẽ", command=set_brush, width=12)
        brush_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        eraser_btn = ttk.Button(tool_buttons, text="Tẩy", command=set_eraser, width=12)
        eraser_btn.pack(side=tk.RIGHT)
        
        # Brush size label
        size_label_frame = ttk.Frame(drawing_section)
        size_label_frame.pack(fill=tk.X, pady=(10, 0), padx=5)
        
        ttk.Label(size_label_frame, text="Kích thước:").pack(side=tk.LEFT)
        self.size_value_label = ttk.Label(size_label_frame, text=str(self.brush_size))
        self.size_value_label.pack(side=tk.RIGHT)
        
        # Brush size slider
        size_frame = ttk.Frame(drawing_section)
        size_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        def update_brush_size(val):
            self.brush_size = int(float(val))
            self.size_value_label.config(text=str(self.brush_size))
        
        self.brush_slider = ttk.Scale(size_frame, from_=1, to=50, orient=tk.HORIZONTAL, 
                                    command=update_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(fill=tk.X)
        
        # Clear mask button
        ttk.Button(drawing_section, text="Xóa vùng đã vẽ", 
                  command=self.clear_mask).pack(fill=tk.X, pady=5, padx=5)
        
        # Processing section
        processing_section = ttk.LabelFrame(left_panel, text="Xử lý")
        processing_section.pack(fill=tk.X, pady=10)
        
        # Poisson blend option
        self.poisson_blend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(processing_section, text="Làm mịn biên giới vùng xử lý", 
                      variable=self.poisson_blend_var).pack(anchor=tk.W, padx=5, pady=5)
        
        # Process button
        self.process_btn = ttk.Button(processing_section, text="Xử lý ảnh",
                                    command=self.process_image)
        self.process_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Save result button
        save_frame = ttk.Frame(processing_section)
        save_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(save_frame, text="Lưu kết quả", 
                  command=self.save_result, width=16).pack(side=tk.LEFT)
                  
        ttk.Button(save_frame, text="Lưu vùng đã vẽ", 
                  command=self.save_mask, width=16).pack(side=tk.RIGHT)
        
        # View mode section
        view_section = ttk.LabelFrame(left_panel, text="Chế độ xem")
        view_section.pack(fill=tk.X, pady=10)
        
        self.view_mode = tk.StringVar(value="split")
        
        modes = [
            ("Trước/Sau", "split"),
            ("Hiển thị vùng vẽ", "overlay"),
            ("Ba bảng riêng biệt", "normal")
        ]
        
        for text, mode in modes:
            ttk.Radiobutton(view_section, text=text, variable=self.view_mode, 
                          value=mode, command=self.update_display).pack(anchor=tk.W, padx=5, pady=2)
        
        # Setup image display area
        display_frame = ttk.Frame(right_panel)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 6))
        self.fig.subplots_adjust(wspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)
        
        # Make room for titles
        plt.tight_layout()
        
        # Initialize empty plots
        for ax in self.axes:
            ax.axis('off')
        
        self.axes[0].set_title("Ảnh gốc - Nhấp và kéo để vẽ", fontsize=11, pad=2)
        self.axes[1].set_title("Kết quả", fontsize=11, pad=2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect events for drawing
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Progress bar
        progress_frame = ttk.Frame(right_panel)
        progress_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Sẵn sàng. Hãy tải ảnh lên để bắt đầu.")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=2)
        
    def select_model(self):
        """Select a model checkpoint file"""
        model_path = filedialog.askopenfilename(
            title="Chọn mô hình",
            filetypes=[("PyTorch Models", "*.pth"), ("Tất cả", "*.*")]
        )
        
        if not model_path:
            return
            
        self.model_path = model_path
        self.model_label.config(text="Mô hình: " + os.path.basename(model_path))
        
        # Load the model in a separate thread
        self.model = None
        threading.Thread(target=self.load_model, daemon=True).start()
    
    def load_image(self):
        """Load an image from file"""
        image_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Tập tin ảnh", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not image_path:
            return
            
        try:
            # Load image
            self.image = Image.open(image_path).convert('RGB')
            
            # Resize if too large
            width, height = self.image.size
            max_size = 1024
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
            
            self.status_var.set(f"Đã tải ảnh: {os.path.basename(image_path)} ({self.image.width}x{self.image.height})")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh: {e}")
    
    def clear_mask(self):
        """Clear the mask"""
        if self.image:
            self.mask = Image.new('L', self.image.size, 0)
            self.mask_draw = ImageDraw.Draw(self.mask)
            self.update_display()
            self.status_var.set("Đã xóa vùng vẽ")
            
    def save_mask(self):
        """Save the current mask to a file"""
        if self.mask is None:
            messagebox.showinfo("Thông báo", "Chưa có vùng vẽ để lưu.")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Lưu vùng vẽ",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("Tất cả", "*.*")]
        )
        
        if save_path:
            try:
                self.mask.save(save_path)
                self.status_var.set(f"Đã lưu vùng vẽ: {os.path.basename(save_path)}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu vùng vẽ: {e}")
    
    def save_result(self):
        """Save the result to file"""
        if self.result is None:
            messagebox.showinfo("Thông báo", "Chưa có kết quả để lưu. Hãy xử lý ảnh trước.")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Lưu kết quả",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        
        if save_path:
            try:
                # Convert result to image and save
                result_img = Image.fromarray((self.result * 255).astype(np.uint8))
                result_img.save(save_path)
                self.status_var.set(f"Đã lưu kết quả: {os.path.basename(save_path)}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu kết quả: {e}")
    
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
        
        # Choose display mode
        mode = self.view_mode.get()
        
        if mode == "normal":
            self._update_normal_view()
        elif mode == "split":
            self._update_split_view()
        else:  # "overlay"
            self._update_overlay_view()
        
        # Update canvas
        self.canvas.draw()
    
    def _update_normal_view(self):
        """Update with three-panel view"""
        # Set up axes
        for i in range(2):
            self.axes[i].clear()
            self.axes[i].axis('off')
        
        # Display original image
        if self.image:
            self.axes[0].imshow(np.array(self.image))
            self.axes[0].set_title("Ảnh gốc - Nhấp và kéo để vẽ", fontsize=11, pad=2)
        
        # Create a composite second panel
        if self.image:
            # Create a figure with two subplots side by side
            if self.result is not None:
                # Show result if available
                self.axes[1].imshow(self.result)
                self.axes[1].set_title("Kết quả xử lý", fontsize=11, pad=2)
            else:
                # Show mask if result not available
                if self.mask:
                    mask_array = np.array(self.mask)
                    self.axes[1].imshow(mask_array, cmap='gray')
                    self.axes[1].set_title("Vùng vẽ", fontsize=11, pad=2)
    
    def _update_split_view(self):
        """Update with before/after split view"""
        if self.image is None:
            return
        
        # Display original in first panel with mask overlay
        self.axes[0].imshow(np.array(self.image))
        
        if self.mask is not None:
            # Create a colored mask for overlay
            mask_array = np.array(self.mask)
            if mask_array.max() > 0:  # Only create overlay if mask has content
                colored_mask = np.zeros((*mask_array.shape, 4), dtype=np.float32)
                colored_mask[:, :, 0] = 1.0  # Red
                colored_mask[:, :, 3] = mask_array / 255.0 * 0.5  # Alpha (50% of mask value)
                self.axes[0].imshow(colored_mask)
        
        self.axes[0].set_title("Ảnh gốc và vùng vẽ", fontsize=11, pad=2)
        
        # Display result in second panel if available
        if self.result is not None:
            self.axes[1].imshow(self.result)
            self.axes[1].set_title("Kết quả xử lý", fontsize=11, pad=2)
        elif self.image is not None:
            # If no result, just show original in second panel
            self.axes[1].imshow(np.array(self.image))
            self.axes[1].set_title("Chưa xử lý", fontsize=11, pad=2)
    
    def _update_overlay_view(self):
        """Update with mask overlay view"""
        if self.image is None:
            return
        
        # Display original in first panel
        self.axes[0].imshow(np.array(self.image))
        self.axes[0].set_title("Ảnh gốc - Nhấp và kéo để vẽ", fontsize=11, pad=2)
        
        # Display original with mask overlay in second panel
        if self.image is not None and self.mask is not None:
            self.axes[1].imshow(np.array(self.image))
            
            # Create a colored mask for overlay
            mask_array = np.array(self.mask)
            if mask_array.max() > 0:  # Only create overlay if mask has content
                colored_mask = np.zeros((*mask_array.shape, 4), dtype=np.float32)
                colored_mask[:, :, 0] = 1.0  # Red
                colored_mask[:, :, 3] = mask_array / 255.0 * 0.5  # Alpha (50% of mask value)
                self.axes[1].imshow(colored_mask)
                self.axes[1].set_title("Chi tiết vùng đã vẽ", fontsize=11, pad=2)
            else:
                self.axes[1].set_title("Chưa vẽ vùng cần xử lý", fontsize=11, pad=2)
    
    def process_image(self):
        """Process the image using the model"""
        if not self.image:
            messagebox.showinfo("Thông báo", "Vui lòng tải ảnh trước.")
            return
        
        if not self.model:
            if not self.model_path or not os.path.exists(self.model_path):
                messagebox.showinfo("Thông báo", "Vui lòng chọn một mô hình hợp lệ.")
                return
            messagebox.showinfo("Thông báo", "Mô hình đang tải. Vui lòng đợi và thử lại.")
            return
        
        # Check if mask is empty
        mask_array = np.array(self.mask)
        if mask_array.max() == 0:
            messagebox.showinfo("Thông báo", "Vui lòng vẽ vùng cần làm nét trước.")
            return
        
        # Update status
        self.status_var.set("Đang xử lý ảnh...")
        self.progress_var.set(10)
        
        # Disable process button while processing
        self.process_btn.config(state=tk.DISABLED)
        
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
            self.root.after(0, lambda: self.status_var.set("Đang chạy mô hình..."))
            with torch.no_grad():
                output = self.model(input_tensor, mask_tensor)
            
            # Convert to numpy for processing
            self.root.after(0, lambda: self.progress_var.set(80))
            input_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()
            output_np = output[0].cpu().permute(1, 2, 0).numpy()
            mask_np = mask_tensor[0].cpu().squeeze().numpy()
            
            if self.poisson_blend_var.get():
                self.root.after(0, lambda: self.status_var.set("Đang làm mịn biên giới..."))
                # Apply Poisson blending
                self.result = poisson_blend(input_np, output_np, mask_np)
            else:
                # Simple alpha blending
                alpha = mask_np[:, :, np.newaxis]
                self.result = input_np * (1 - alpha) + output_np * alpha
            
            # Update UI
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, self._update_after_processing)
        except Exception as e:
            print(f"Lỗi khi xử lý: {e}")
            self.root.after(0, lambda: self.status_var.set(f"Lỗi: {e}"))
            self.root.after(0, lambda: self.progress_var.set(0))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
    
    def _update_after_processing(self):
        """Update UI after processing is complete"""
        self.update_display()
        self.status_var.set("Xử lý hoàn tất")
        self.process_btn.config(state=tk.NORMAL)
        # Reset progress bar after a delay
        self.root.after(1000, lambda: self.progress_var.set(0))
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Giao diện làm nét hình ảnh đơn giản')
    parser.add_argument('--model_path', type=str, default=None, help='Đường dẫn đến mô hình')
    parser.add_argument('--device', type=str, default='cuda', help='Thiết bị sử dụng (cuda hoặc cpu)')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    app = SimpleImageSharpener(model_path=args.model_path, device=args.device)
    app.run()

if __name__ == '__main__':
    main() 