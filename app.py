import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import io

class ArtisticStyleTransfer:
    def __init__(self, image):
        self.image = image
        if self.image is None:
            raise ValueError("Invalid image")

    def oil_painting(self, size=10, dyn_ratio=1):
        # In newer OpenCV versions, oilPainting requires integer dynRatio
        # and may have different parameter requirements
        try:
            # Try first with standard parameters (newer versions)
            return cv2.xphoto.oilPainting(self.image, size, int(dyn_ratio))
        except Exception as e:
            try:
                # For even newer versions that may require different parameters
                return cv2.stylization(self.image, sigma_s=60, sigma_r=0.45)
            except Exception as e2:
                # Fallback using alternative approach if both fail
                img = self.image.copy()
                # Simplified oil painting effect using bilateral filter
                for _ in range(3):
                    img = cv2.bilateralFilter(img, 9, 75, 75)
                return img

    def watercolor(self):
        img = self.image.copy()
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        for _ in range(3):
            img = cv2.medianBlur(img, 3)
        img = cv2.edgePreservingFilter(img, flags=1, sigma_s=5, sigma_r=0.4)
        img = cv2.bilateralFilter(img, d=3, sigmaColor=10, sigmaSpace=5)
        for _ in range(2):
            img = cv2.bilateralFilter(img, d=3, sigmaColor=20, sigmaSpace=10)
        for _ in range(3):
            img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=10)
        img = cv2.GaussianBlur(img, (7,7), sigmaX=2, sigmaY=2)
        sharpened = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (0,0), 3), -0.5, 0)
        sharpened = cv2.addWeighted(sharpened, 1.4, cv2.GaussianBlur(sharpened, (0,0), 2), -0.2, 10)
        return sharpened

    def cartoon(self):
        img = self.image.copy()
        quantized = cv2.pyrMeanShiftFiltering(img, sp=20, sr=40)
        gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        cartoon_img = quantized.copy()
        cartoon_img[edges != 0] = [0,0,0]
        return cartoon_img

    def frequency_low_pass(self, sigma=10):
        img = self.image.astype(np.float32)
        result = np.zeros_like(img)
        for channel in range(3):
            f = np.fft.fft2(img[:, :, channel])
            fshift = np.fft.fftshift(f)
            rows, cols = img.shape[:2]
            crow, ccol = rows // 2, cols // 2
            x = np.arange(cols) - ccol
            y = np.arange(rows) - crow
            X, Y = np.meshgrid(x, y)
            gaussian = np.exp(-(X**2 + Y**2)/(2*sigma**2))
            fshift_filtered = fshift * gaussian
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            result[:, :, channel] = img_back
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def color_quantization(self, k=8):
        img = self.image.copy()
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(img.shape)
        return quantized

    def enhance_saturation(self, factor=1.5):
        img = self.image.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, factor)
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        return enhanced
        
    # New filters below
    
    def ghibli_style(self):
        """
        Create a Studio Ghibli inspired filter with soft colors, 
        enhanced contrast and a dreamy atmosphere
        """
        img = self.image.copy()
        
        # Step 1: Soften the image while preserving edges
        img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
        
        # Step 2: Enhance colors (Ghibli's vibrant but natural palette)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Slightly boost saturation for vibrant colors
        s = cv2.multiply(s, 1.3)
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        # Enhance contrast in luminance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        
        hsv_enhanced = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Step 3: Apply subtle color grading (warm highlights, cool shadows)
        # Split channels
        b, g, r = cv2.split(img)
        
        # Enhance blue in shadows slightly
        b_adjusted = np.where(v < 100, np.minimum(b * 1.1, 255), b)
        
        # Enhance yellow/red in highlights
        r_adjusted = np.where(v > 150, np.minimum(r * 1.1, 255), r)
        g_adjusted = np.where(v > 150, np.minimum(g * 1.05, 255), g)
        
        img = cv2.merge([b_adjusted.astype(np.uint8), 
                         g_adjusted.astype(np.uint8), 
                         r_adjusted.astype(np.uint8)])
        
        # Step 4: Final subtle blur for dreamy effect
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        # Step 5: Subtle vignette effect common in Ghibli films
        rows, cols = img.shape[:2]
        
        # Generate vignette mask
        X_center = cols / 2
        Y_center = rows / 2
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Calculate distance from center
        dist = np.sqrt((X - X_center) ** 2 + (Y - Y_center) ** 2)
        
        # Normalize distance
        dist = dist / np.max(dist)
        
        # Create vignette mask (very subtle)
        vignette = 1 - dist * 0.2
        vignette = np.dstack([vignette] * 3)
        
        # Apply vignette
        img = np.multiply(img, vignette).astype(np.uint8)
        
        return img

    def noir_filter(self):
        """
        Create a film noir style with high contrast black and white,
        with emphasis on shadows and highlights
        """
        img = self.image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Increase global contrast
        alpha = 1.5  # Contrast control
        beta = -30   # Brightness control
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Add film grain
        noise = np.zeros(gray.shape, np.uint8)
        cv2.randn(noise, 0, 15)  # mean=0, stddev=15
        gray = cv2.add(gray, noise)
        
        # Apply vignette effect
        rows, cols = gray.shape
        
        # Generate vignette mask
        X_center = cols / 2
        Y_center = rows / 2
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Calculate distance from center
        dist = np.sqrt((X - X_center) ** 2 + (Y - Y_center) ** 2)
        
        # Normalize distance
        dist = dist / np.max(dist)
        
        # Create vignette mask (stronger for noir)
        vignette = 1 - dist * 0.5
        
        # Apply vignette
        gray = np.multiply(gray, vignette).astype(np.uint8)
        
        # Convert back to BGR for consistency with other filters
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def cyberpunk(self, purple_strength=1.8, teal_strength=1.5):
        """
        Create a cyberpunk-inspired filter with neon highlights,
        purple and teal color grading
        """
        img = self.image.copy()
        
        # Step 1: Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Step 2: Split channels for color grading
        b, g, r = cv2.split(img)
        
        # Step 3: Enhance blues and purples in shadows
        # Boost blue channel
        b = np.minimum(b * purple_strength, 255).astype(np.uint8)
        
        # Step 4: Enhance teals (blue+green) in midtones
        g = np.minimum(g * teal_strength, 255).astype(np.uint8)
        
        # Step 5: Recombine
        img = cv2.merge([b, g, r])
        
        # Step 6: Add "digital" feel with slight posterization
        levels = 5
        img = (img // (256 // levels)) * (256 // levels)
        
        # Step 7: Add "glow" to bright areas
        _, bright_areas = cv2.threshold(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
            200, 255, 
            cv2.THRESH_BINARY
        )
        
        # Dilate bright areas to create glow effect
        kernel = np.ones((5, 5), np.uint8)
        bright_areas = cv2.dilate(bright_areas, kernel, iterations=3)
        bright_mask = cv2.GaussianBlur(bright_areas, (21, 21), 0)
        
        # Create glow layer (neon purple)
        glow = np.zeros_like(img)
        glow[bright_mask > 0] = [255, 100, 255]  # Purple glow
        glow = cv2.GaussianBlur(glow, (21, 21), 0)
        
        # Blend with original
        result = cv2.addWeighted(img, 0.8, glow, 0.2, 0)
        
        return result

    def vintage(self):
        """
        Create a vintage/retro filter with faded colors, 
        slight sepia tone, and vignette
        """
        img = self.image.copy()
        
        # Step 1: Create slight sepia effect
        sepia_kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        
        sepia = cv2.transform(img, sepia_kernel)
        
        # Step 2: Reduce saturation
        hsv = cv2.cvtColor(sepia, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 0.7)  # Reduce saturation
        hsv_faded = cv2.merge([h, s, v])
        faded = cv2.cvtColor(hsv_faded, cv2.COLOR_HSV2BGR)
        
        # Step 3: Add slight yellow/orange tint - fixing the type issue
        tint = np.ones_like(faded) * np.array([30, 50, 80], dtype=np.uint8)
        faded = cv2.addWeighted(faded, 0.9, tint, 0.1, 0)
        
        # Step 4: Lower contrast
        faded = cv2.convertScaleAbs(faded, alpha=0.8, beta=20)
        
        # Step 5: Add noise/grain
        noise = np.zeros_like(faded)
        cv2.randn(noise, 0, 8)
        faded = cv2.add(faded, noise)
        
        # Step 6: Add vignette
        rows, cols = faded.shape[:2]
        
        # Create vignette mask
        X_center = cols / 2
        Y_center = rows / 2
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Calculate distance from center
        dist = np.sqrt((X - X_center) ** 2 + (Y - Y_center) ** 2)
        
        # Normalize distance
        dist = dist / np.max(dist)
        
        # Create vignette mask
        vignette = 1 - dist * 0.35
        vignette = np.dstack([vignette] * 3)
        
        # Apply vignette
        faded = np.multiply(faded, vignette).astype(np.uint8)
        
        return faded

    def impressionist(self):
        """
        Create an impressionist painting style filter
        with brush-like strokes and enhanced colors
        """
        img = self.image.copy()
        
        # Step 1: Enhance colors slightly
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv_enhanced = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Step 2: Apply edge-preserving filter for painterly effect
        img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.8)
        
        # Step 3: Create brush stroke effect with multiple iterations of bilateral filter
        for _ in range(2):
            img = cv2.bilateralFilter(img, d=9, sigmaColor=150, sigmaSpace=150)
        
        # Step 4: Create detailed strokes
        for radius in [3, 5, 7]:
            temp = img.copy()
            temp = cv2.GaussianBlur(temp, (0, 0), radius)
            img = cv2.addWeighted(img, 0.6, temp, 0.4, 0)
        
        # Step 5: Enhance image luminance variations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detail = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        detail = cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)
        
        # Add subtle details back
        img = cv2.addWeighted(img, 0.9, detail, 0.1, 0)
        
        # Step 6: Final color enhancement
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        v = clahe.apply(v)
        hsv_enhanced = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        return img

    def pixel_sort(self, threshold=30, vertical=False):
        """
        Create a glitch art effect via pixel sorting
        """
        img = self.image.copy()
        if vertical:
            img = cv2.transpose(img)
        
        # Convert to grayscale for finding sorting regions
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find edges
        edges = cv2.Canny(gray, threshold, threshold * 2)
        
        # Process each row
        for y in range(img.shape[0]):
            intervals = []
            start = None
            
            # Find intervals to sort
            for x in range(img.shape[1]):
                if edges[y, x] == 0 and start is None:
                    start = x
                elif edges[y, x] > 0 and start is not None:
                    intervals.append((start, x))
                    start = None
            
            # Add the last interval if needed
            if start is not None:
                intervals.append((start, img.shape[1]))
            
            # Sort pixels in each interval
            for start, end in intervals:
                if end - start < 2:
                    continue
                    
                # Extract interval
                interval = img[y, start:end].copy()
                
                # Sort by brightness
                brightness = np.sum(interval, axis=1)
                sort_indices = np.argsort(brightness)
                
                # Apply sorted pixels
                img[y, start:end] = interval[sort_indices]
        
        if vertical:
            img = cv2.transpose(img)
            
        return img

    def duotone(self, dark_color=(0, 51, 76), light_color=(255, 221, 149)):
        """
        Create a duotone effect using two colors
        """
        img = self.image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Create an empty colored image
        duotone = np.zeros_like(img)
        
        # Map grayscale values to a gradient between the two colors
        for i in range(3):  # For each channel
            # Create a gradient from dark to light color for this channel
            channel_gradient = np.interp(gray, 
                                         [0, 255], 
                                         [dark_color[i], light_color[i]])
            duotone[:, :, i] = channel_gradient
            
        return duotone


# Streamlit app
def main():
    st.set_page_config(
        page_title="Artistic Style Transfer",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Artistic Style Transfer")
    st.subheader("Transform your photos into artistic styles!")

    # Create sidebar for options
    st.sidebar.title("Style Options")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # List of available styles
    styles = {
        "Original": None,
        "Oil Painting": "oil_painting",
        "Watercolor": "watercolor",
        "Cartoon": "cartoon",
        "Ghibli Style": "ghibli_style",
        "Film Noir": "noir_filter",
        "Cyberpunk": "cyberpunk",
        "Vintage": "vintage",
        "Impressionist": "impressionist",
        "Pixel Sort": "pixel_sort",
        "Vertical Pixel Sort": "vertical_pixel_sort",
        "Low Pass Filter": "frequency_low_pass",
        "Color Quantization": "color_quantization",
        "Enhanced Saturation": "enhance_saturation",
        "Duotone": "duotone"
    }
    
    # Style selection
    selected_style = st.sidebar.selectbox(
        "Choose a style",
        list(styles.keys())
    )
    
    # Parameters for specific filters
    if selected_style == "Oil Painting":
        size = st.sidebar.slider("Brush Size", 1, 20, 10)
        dyn_ratio = st.sidebar.slider("Dynamic Ratio", 0.1, 5.0, 1.0)
    
    elif selected_style == "Color Quantization":
        k = st.sidebar.slider("Number of Colors", 2, 32, 8)
    
    elif selected_style == "Enhanced Saturation":
        factor = st.sidebar.slider("Saturation Factor", 0.5, 3.0, 1.5)
    
    elif selected_style == "Low Pass Filter":
        sigma = st.sidebar.slider("Sigma (Blurriness)", 1, 50, 10)
    
    elif selected_style == "Cyberpunk":
        purple_strength = st.sidebar.slider("Purple Strength", 1.0, 3.0, 1.8)
        teal_strength = st.sidebar.slider("Teal Strength", 1.0, 3.0, 1.5)
    
    elif selected_style == "Pixel Sort" or selected_style == "Vertical Pixel Sort":
        threshold = st.sidebar.slider("Threshold", 10, 100, 30)
    
    elif selected_style == "Duotone":
        color_presets = {
            "Blue & Yellow": ((0, 51, 76), (255, 221, 149)),
            "Purple & Yellow": ((75, 0, 130), (255, 255, 0)),
            "Red & Teal": ((165, 30, 30), (0, 128, 128)),
            "Green & Pink": ((0, 100, 0), (255, 182, 193)),
            "Black & White": ((0, 0, 0), (255, 255, 255)),
            "Orange & Blue": ((255, 128, 0), (0, 128, 255)),
            "Custom": None
        }
        
        preset = st.sidebar.selectbox("Color Preset", list(color_presets.keys()))
        
        if preset == "Custom":
            dark_color = st.sidebar.color_picker("Dark Color", "#003366")
            light_color = st.sidebar.color_picker("Light Color", "#ffdd95")
            
            # Convert hex to BGR
            dark_r, dark_g, dark_b = int(dark_color[1:3], 16), int(dark_color[3:5], 16), int(dark_color[5:7], 16)
            light_r, light_g, light_b = int(light_color[1:3], 16), int(light_color[3:5], 16), int(light_color[5:7], 16)
            
            dark_color_bgr = (dark_b, dark_g, dark_r)
            light_color_bgr = (light_b, light_g, light_r)
        else:
            dark_color_bgr, light_color_bgr = color_presets[preset]
    
    # Download option
    download_format = st.sidebar.selectbox(
        "Download Format",
        ["JPG", "PNG"]
    )
    
    # Image display column
    col1, col2 = st.columns(2)
    
    # Process image when uploaded
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Create style transfer object
        ast = ArtisticStyleTransfer(img)
        
        # Display original image
        with col1:
            st.header("Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Apply selected style
        with col2:
            st.header(f"{selected_style}")
            
            try:
                if selected_style == "Original":
                    styled_img = img
                elif selected_style == "Oil Painting":
                    styled_img = ast.oil_painting(size=size, dyn_ratio=dyn_ratio)
                elif selected_style == "Watercolor":
                    styled_img = ast.watercolor()
                elif selected_style == "Cartoon":
                    styled_img = ast.cartoon()
                elif selected_style == "Ghibli Style":
                    styled_img = ast.ghibli_style()
                elif selected_style == "Film Noir":
                    styled_img = ast.noir_filter()
                elif selected_style == "Cyberpunk":
                    styled_img = ast.cyberpunk(purple_strength=purple_strength, teal_strength=teal_strength)
                elif selected_style == "Vintage":
                    styled_img = ast.vintage()
                elif selected_style == "Impressionist":
                    styled_img = ast.impressionist()
                elif selected_style == "Pixel Sort":
                    styled_img = ast.pixel_sort(threshold=threshold, vertical=False)
                elif selected_style == "Vertical Pixel Sort":
                    styled_img = ast.pixel_sort(threshold=threshold, vertical=True)
                elif selected_style == "Low Pass Filter":
                    styled_img = ast.frequency_low_pass(sigma=sigma)
                elif selected_style == "Color Quantization":
                    styled_img = ast.color_quantization(k=k)
                elif selected_style == "Enhanced Saturation":
                    styled_img = ast.enhance_saturation(factor=factor)
                elif selected_style == "Duotone":
                    if preset == "Custom":
                        styled_img = ast.duotone(dark_color=dark_color_bgr, light_color=light_color_bgr)
                    else:
                        styled_img = ast.duotone(dark_color=dark_color_bgr, light_color=light_color_bgr)
                
                # Display styled image
                st.image(cv2.cvtColor(styled_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Download button
                if styled_img is not None:
                    # Convert to PIL Image
                    pil_img = Image.fromarray(cv2.cvtColor(styled_img, cv2.COLOR_BGR2RGB))
                    
                    # Create buffer for downloading
                    buf = io.BytesIO()
                    if download_format == "PNG":
                        pil_img.save(buf, format="PNG")
                        mime_type = "image/png"
                        file_ext = "png"
                    else:
                        pil_img.save(buf, format="JPEG", quality=95)
                        mime_type = "image/jpeg"
                        file_ext = "jpg"
                    
                    buf.seek(0)
                    
                    # Download button
                    st.download_button(
                        label="Download Styled Image",
                        data=buf,
                        file_name=f"styled_image_{selected_style.lower().replace(' ', '_')}.{file_ext}",
                        mime=mime_type
                    )
            
            except Exception as e:
                st.error(f"Error applying style: {str(e)}")
    else:
        # Display instructions
        st.info("👈 Please upload an image to get started!")
        
        # Example images
        with st.expander("See examples"):
            st.markdown("""
            Here are some examples of what our filters can do:
            
            - **Ghibli Style**: Soft, vibrant colors with dreamy atmosphere
            - **Cyberpunk**: Neon highlights with purple and teal color grading
            - **Film Noir**: High-contrast black and white with dramatic shadows
            - **Vintage**: Retro look with faded colors and sepia toning
            - **Pixel Sort**: Glitch art effect with sorted pixels for a unique look
            """)

if __name__ == "__main__":
    main()