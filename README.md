# AxxTerm

Serial terminal with dual ASCII/HEX view, real-time plotting, data converter, and configurable macro buttons.

![AxxTerm GUI](AxxTerm_GUI.PNG)

## Features

- **Dual data view** - ASCII and HEX side by side, color-coded (red = received, blue = sent)
- **Real-time plotting** - Graph incoming data with 1-12 channels, auto-scaling Y axis, and adjustable plot length. Supports ASCII text, binary stream, and custom frame decoding modes
- **Send modes** - ASCII, HEX, and Binary with configurable line endings (LF, CR, CRLF)
- **Macro buttons** - 8 quick-send buttons, right-click to edit label and payload
- **Data converter** - Convert between HEX, ASCII, Decimal, and Binary
- **Serial configuration** - Baud rate, data bits, parity, stop bits, flow control

## Requirements

- Python 3.8+
- PyQt5
- pyqtgraph
- NumPy

## Installation

### Run from source

```bash
pip install -r requirements.txt
python AxxTerm_serial.py
```

### Build standalone .exe (no Python needed on target machine)

```bash
build.bat
```

This installs PyInstaller if needed, then creates `dist\AxxTerm.exe` - a single portable executable. Copy it anywhere and run.

You can also build manually:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name AxxTerm --clean AxxTerm_serial.py
```

## Usage

### Connecting

1. Select the COM port from the dropdown (click **Scan** to refresh the list)
2. Set baud rate (default 115200) and serial parameters (data bits, parity, stop bits, flow control)
3. Click **Open** to connect - the DB-9 connector icon turns green when connected

### Sending data

- Type in the input field and press **Enter** or click **Send**
- Select **ASCII**, **HEX**, or **BINARY** mode from the dropdown on the left
- Select line ending from the dropdown on the right (LF, CR, CRLF, or none)
- Use **Up/Down arrows** to navigate send history

### Macro buttons

The 8 green buttons at the bottom send pre-configured hex data with a single click.

- **Right-click** any macro button to edit it
- Set a custom label and payload using HEX, ASCII, Decimal, or Binary input
- Macros are saved automatically to `macros.json` next to the application

### Real-time plotting

1. Check **Show Graph** to enable the plot
2. Set the number of channels with the **Ch** spinner (1-12)
3. Set the scrolling window size with the **Pts** spinner
4. Select the data mode from the **Mode** dropdown

#### Data modes

**ASCII** (default) - Parses incoming text lines. Supports several formats:

| Format | Example |
|--------|---------|
| Tab-separated | `1.0\t2.0\t3.0` |
| Comma-separated | `1.0,2.0,3.0` |
| Space-separated | `1.0 2.0 3.0` |
| Labeled | `temp:23.5\thum:45.2\tpres:1013` |

Each line (terminated by `\n`) is parsed and plotted. Values beyond the configured channel count are ignored.

**Binary Stream** - Decodes a continuous stream of raw binary data. Configure the data type (uint8, int8, uint16, int16, uint32, int32, float32, double64) and byte order (little/big endian). Each sample is `channels x sizeof(type)` bytes. Use the **Sync** button to re-align if the stream gets out of phase.

**Custom Frame** - Decodes framed binary packets with the structure: `[Sync Word] [Optional Size] [Payload] [Optional Checksum]`. Configure:
- **Sync Word** - hex bytes to match at frame start (e.g., `AA BB`)
- **Size Field** - Fixed (known size), 1-byte, or 2-byte length field after sync
- **Frame Size** - payload size in bytes (when using Fixed size field)
- **Checksum** - optional 8-bit sum validation

In Binary Stream and Custom Frame modes, the left panel shows color-coded decoded channel values and the right panel shows the raw hex dump. Settings are saved automatically to `plot_settings.json`.

### Data converter

The converter at the bottom converts between HEX, ASCII, Decimal, and Binary in real time. Select the conversion type from the dropdown and type in the left field - the result appears in the right field. Changing the conversion type re-converts the current input automatically.

## Files

| File | Description |
|------|-------------|
| `AxxTerm_serial.py` | Main application source |
| `requirements.txt` | Python dependencies |
| `build.bat` | Build script for standalone .exe |
| `macros.json` | Macro button config (auto-created on first edit) |
| `plot_settings.json` | Plot/decode settings (auto-created on change) |
| `test_readers.py` | Unit tests for binary/frame reader classes |

## License

See [LICENSE](LICENSE) for details.
