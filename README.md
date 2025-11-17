# Integrated Gradients for Vision Models

An implementation of Integrated Gradients (IG) for interpreting image classification models, specifically designed for Vision Transformers (ViT) and Convolutional Neural Networks.

## What are Integrated Gradients?

Integrated Gradients is an attribution method that explains predictions of deep neural networks by attributing the prediction to input features. The key idea is to compute the gradients of the model's output with respect to the input along a straight path from a baseline (e.g., a black image) to the actual input.

Mathematically, for an input image **x** and baseline **x'**, the integrated gradient for feature *i* is:

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

Where:
- **F** is the model's output (logit) for a specific class
- **α** is the interpolation parameter (0 = baseline, 1 = actual input)
- The integral is approximated using Riemann sum

This method satisfies desirable properties like **sensitivity** (if an input feature changes the output, it receives non-zero attribution) and **implementation invariance** (functionally equivalent models receive identical attributions).

## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Apple Silicon Macs, ensure you have MPS support enabled in PyTorch.

## Output

The function returns a list of 3 numpy arrays (one for each of the top-3 predicted classes), where each array has shape `(C, H, W)` representing the attribution for each pixel.

The function also displays:
1. **Top 3 predictions** with their logit values
2. **Attribution heatmaps** overlaid on the original image for each class, showing which pixels were most important for each prediction

## Visualization

The visualization uses:
- **Original image** as the background
- **Turbo colormap** for attribution magnitude (bright = high importance)
- **Alpha blending** to overlay attributions on the image

Attribution values are:
- Aggregated across color channels (summing absolute values)
- Normalized to [0, 1] for visualization
- Displayed for all top-3 predicted classes

## Supported Models

### ConvNeXt (CNN)
- Model: `facebook/convnext-base-224`
- 224×224 input resolution
- Modern CNN architecture with competitive performance

### Vision Transformer (ViT)
- Model: `google/vit-base-patch16-224`
- 224×224 input resolution  
- Attention-based architecture

## Choosing Baselines

The choice of baseline can significantly affect attributions:

- **Black baseline** (`baseline='black'`): Good for natural images where absence = darkness
- **White baseline** (`baseline='white'`): Useful for inverted images or medical imaging
- **Noise baseline** (`baseline='noise'`): Helps identify robust features

For most natural images, black baseline is recommended.

## Example Results
```python
# Analyze a dog image
ig('golden_retriever.jpg', model='can', num_alpha_steps=25)
```

Output:
```
Using black baseline and can model
Using device: mps

Top 3 classes:
golden retriever: 15.23
Labrador retriever: 12.45
cocker spaniel: 10.87

Computing IG for class 1: golden retriever
[Progress bar...]
```

The visualization will show which parts of the image (e.g., face, fur texture) were most important for each prediction.

## Technical Notes

- **Device**: Automatically uses MPS (Apple Silicon) or CPU
- **Memory**: Gradients are computed for each interpolation step; increase `num_alpha_steps` for better approximation but higher memory usage
- **Performance**: Processing time scales with `num_alphas × num_alpha_steps`

## References

- [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) (Sundararajan et al., 2017)
- [Integrated Gradients for Importance Attribution](https://distill.pub/2020/attribution-baselines/) (Distill, 2020)

## License

MIT

## Author

Paul Savala
St. Edward's University  
www.github.com/paulsavala
