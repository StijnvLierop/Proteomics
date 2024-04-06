# Define body fluid names
BODY_FLUIDS = ["saliva", "semen", "vaginalfluid", "urine", "blood"]

# Define color mapping for visualizations
COLOR_MAPPING = {'blood': '#FF0000',
                 'saliva': 'lightblue',
                 'vaginalfluid': 'green',
                 'urine': 'orange',
                 'semen': 'darkblue'}

# Plot download config
plot_download_config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
    'height': 1000,
    'width': 1000,
    'scale':6 # Multiply title/legend/axis/canvas sizes by this factor
  }
}
