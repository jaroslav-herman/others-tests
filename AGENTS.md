# Project preferences

- For electrochemical measurement data, prefer `.mpr` files by default. Use other file formats only when explicitly requested or when no suitable `.mpr` files are available.
- For requests such as “Show me polarization curves for sample XY”, use `show_455_iv_curves.py` as the reference workflow: discover `.mpr` files with `we.load_files`, read them with `we.read_file_safe`, extract curves with `wepy.iv_curve.IV_curves_data`, and color the time evolution with `we.get_colors`.
- Keep the default colorscale in `we.get_colors`; do not override its colormap unless explicitly requested. Use `we.get_sample_name(sample_number, sample_folder / "sample_log.csv")` to label plots from `Sample Name` in each sample's folder.
- For graph-producing requests, run the generated Python script after creating it so the requested graph files are produced and validated.
- For requests such as “compare performance time evolution for some samples”, use `compare_performance_evolution_453_455_457.py` as the reference pattern: use `we.load_folders`, process preferred `.mpr` files with `we.read_file_safe`, extract IV curves with `wepy.iv_curve.IV_curves_data`, read per-sample names from each folder's `sample_log.csv`, preserve the default `we.get_colors()` scale, and create separate graphs for each requested cell voltage.
- For plot labels, use the shared Google Sheet sample table as the primary source for sample names. If a matching `Sample Name` is empty or unavailable, use only the sample number.
- Use the sample `Type` from the live Google Sheet for folder selection: `AEM`
  samples are under the year's `AEM-WE` subfolder; other known types are under
  the year folder; if `Type` is blank or unavailable, search both locations.
- Treat FTACV work in this project as Python-based Fourier Transform Alternating
  Current cyclic voltammetry analysis. Prefer the project's data-processing
  workflow for loading, cleaning, transforming, filtering, validating, and
  plotting electrochemical time-series data, with `.mpr` as the default source
  format.

- For general electrochemical plotting requests, use wepy.basics.read_file() to load .mpr or .mpt files, extract data columns with automatic handling of <I>/mA vs I/mA naming variations, use wepy.get_colors() for multi-series color consistency, and create project subfolders inside others-tests by default unless a specific output location is requested.

- When creating or editing files (especially images), provide direct file paths in the response so users can instantly access them. Use markdown image syntax for PNG/JPG files: ![description](file:///absolute/path/to/file.png). For other file types, provide the absolute path as a clickable link.

- When providing file links in chat, use the absolute Windows path without file:/// protocol for best compatibility. Format: C:\Users\...\file.png. For images, also include markdown syntax: ![alt](C:\Users\...\file.png). Test links before finalizing response.

- For creating polarization curve plots from SV technique data: use wepy.load_files() to discover .mpr files containing 'SV' in the filename, load them with wepy.read_file(), extract IV curves with wepy.IV_curves_data(data), use wepy.get_colors() for multi-series coloring, plot I/mA vs control/V (or Ecell/V), and save as PNG. Reference: plot_polarization_curve_443_MoS2.py in Summer projects/MoS2.
