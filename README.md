# Cloud Map Generator

## Live Cloud Map

![Live Cloud Map](https://raw.githubusercontent.com/BrianHToro/Pythonclouds/main/output/clouds.jpg)

*This image updates automatically every 3 hours with the most recent global cloud cover data from EUMETSAT satellites.*

---

## Installation

1. Download the project files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The project uses a simple configuration file (`config.json`) for WMS layer settings. No API keys are required as it uses publicly available EUMETSAT WMS services.

## Usage

### Basic Usage

```bash
python main.py
```

### Advanced Usage

```bash
# Specify output directory
python main.py --output-dir ./cloud_maps

# Custom image dimensions
python main.py --width 4096 --height 2048

# Use configuration file
python main.py --config my_config.json
```

### Command Line Options

- `--output-dir`: Output directory for generated images (default: "output")
- `--config`: Path to configuration file
- `--width`: Image width in pixels (default: 8192)
- `--height`: Image height in pixels (default: 4096)

## Output Files

The script generates:

1. **clouds.jpg** - Cloud cover map in 2:1 aspect ratio (overwrites previous image)

## Data Sources

Uses EUMETSAT WMS services for satellite imagery:
- **IR10.8 μm**: Infrared imagery for cloud detection
- **Dust RGB**: Multi-spectral imagery for dust and cloud analysis
- **Natural Colour RGB**: True-color imagery

## Technical Details

### Image Processing Pipeline

1. **Data Fetching**: Retrieve satellite imagery from EUMETSAT WMS services
2. **Preprocessing**: Convert to appropriate format and normalize
3. **Cloud Processing**: Apply advanced cloud detection algorithms using IR, dust, and visible data
4. **Gap Filling**: Interpolate white gaps in stitched imagery
5. **Final Processing**: Apply gamma correction and screen blending

## Examples

### Generate Cloud Map

```bash
python main.py --width 4096 --height 2048
```

### Generate High Resolution Cloud Map

```bash
python main.py --width 8192 --height 4096
```

## Troubleshooting

### Common Issues

1. **Network Timeout**: Check your internet connection and EUMETSAT service status
2. **Memory Issues**: Reduce image dimensions for large images
3. **Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`

### Logging

The script provides detailed logging. Check the console output for:
- Data fetching status
- Image processing progress
- Error messages and warnings

## Attribution

- **EUMETSAT Data**: Contains modified EUMETSAT data

## Contributing

Feel free to submit issues and enhancement requests!

## References

- [EUMETSAT Data Store](https://data.eumetsat.int/)
- [EUMETSAT Product Viewer](https://view.eumetsat.int/productviewer)
