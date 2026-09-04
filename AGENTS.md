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
