# Astyle: Artistic Style Transfer Tool

Astyle is an interactive tool that transforms your photos into various artistic styles, ranging from classic painting techniques to modern digital aesthetics. Built with OpenCV and Streamlit, this application provides an intuitive interface for applying complex image processing filters with just a few clicks.

Try it now: https://astyle.streamlit.app/

## 🌟 Features

- **14 Artistic Filters**: Including Oil Painting, Watercolor, Studio Ghibli, Cyberpunk, Film Noir, and more
- **Customizable Parameters**: Fine-tune each filter with intuitive sliders
- **Real-time Preview**: See transformations immediately
- **High-Quality Exports**: Download your creations in JPG or PNG format
- **User-friendly Interface**: No technical knowledge required

## 🖼️ Filter Showcase

### Studio Ghibli Style
Transform your photos with the distinctive soft, vibrant, and dreamy aesthetic inspired by Studio Ghibli animations.

### Cyberpunk
Create a futuristic nightscape with neon highlights, purple and teal color grading.

### Film Noir
Convert images to dramatic high-contrast black and white with a vintage cinema feel.

### Impressionist
Apply a painting-like effect reminiscent of impressionist art.

### Pixel Sort
Create a glitch art aesthetic by sorting pixels based on brightness.

## 📥 Installation

### Prerequisites
- Python 3.7+

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Astyle
   cd Astyle
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Launch the application:
```bash
streamlit run app.py
```

Your default web browser should automatically open to the application. If not, navigate to the URL shown in the terminal (typically http://localhost:8501).

### Basic Workflow:
1. Upload an image using the file uploader in the sidebar
2. Select a style from the dropdown menu
3. Adjust parameters as desired using the sliders
4. Download your creation using the "Download Styled Image" button

## 🔧 Technical Details

Astyle uses various image processing techniques:
- Bilateral filtering
- Color space transformations
- Frequency domain manipulation
- Edge detection and enhancement
- Color quantization
- Custom convolution operations

## 📚 Code Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Required Python packages
- `examples/`: Sample images showing various filter effects
- `docs/`: Additional documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 Future Plans

- [ ] Add more artistic styles
- [ ] Implement neural style transfer using deep learning
- [ ] Create batch processing capabilities
- [ ] Develop CLI version for automation
- [ ] Optimize performance for larger images

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- OpenCV for image processing capabilities
- Streamlit for the interactive web interface
- The artistic and computational photography community for inspiration

---
