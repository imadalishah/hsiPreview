# HSI Preview
A Python-based dashboard for visualizing and analyzing Hyperspectral Images (HSI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)

## Overview
HSI Preview is a simple yet a quick visualizing and analyzing tool for hyperspectral imaging data across multiple formats. It offers an intuitive interface for researchers, data scientists, and professionals working with hyperspectral data.

![image](https://github.com/user-attachments/assets/84665a88-cbbd-4511-b178-ba9875ac5d82)
<p align="center"><em><strong>HSI Preview Dashboard showing the main interface with spectral visualization, band selection, and analysis tools. Dataset: HSI-Drive v2 [1]</strong></em> </p>


## Key Features
- Support for multiple HSI data formats (.mat, .npy, etc.)
- Flexible dimension handling with visual preview capabilities
- Channel-by-channel cube visualization
- Advanced image enhancement tools:
  - Contrast adjustment
  - Brightness control
  - Orientation alignment
- Spectral reflectance analysis for individual pixels
- Interactive dashboard interface

## Installation

### Prerequisites
- Python 3.x
- Dependencies (installed via requirements.txt)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/imadalishah/hsiPreview.git
   cd hsiPreview


2. Install required packages:
   ```py
   pip install -r requirements.txt


4. Usage
   ```bash
   Launch the dashboard:
   python dashboard.py

## Data Handling

- Supported Formats: .mat, .npy, and other common HSI data formats
- Dimension Management:
- Flexible dimension sequence selection
- Visual preview for dimension verification
- Automated dimension detection

## Contributing
Contributions are welcome! Please feel free to:
- Submit bug reports
- Propose new features
- Create pull requests

## Note:
*This project is currently being revised - expect errors and bugs!!!*

If you find this project helpful, please consider giving it a ⭐️

## References
[1] J. Gutiérrez-Zaballa, K. Basterretxea, J. Echanobe, M. V. Martínez, and U. Martinez-Corral, "HSI-Drive v2.0: More Data for New Challenges in Scene Understanding for Autonomous Driving," in *2023 IEEE Symposium Series on Computational Intelligence (SSCI)*, Dec. 2023, pp. 207-214, IEEE.
