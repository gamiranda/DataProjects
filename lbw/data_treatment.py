import functions as f

df = f.download_sinasc_data(2024)

df = (
    df.pipe(f.remove_unused_features)
      .pipe(f.create_features)
      .pipe(f.optimize_dtypes)
)
