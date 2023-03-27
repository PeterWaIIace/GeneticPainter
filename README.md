# GeneticPainter
My take on genetic painters - algorithm painting reference image using genetic algorithms with crossover and mutations.

All results are achieved by pure crossover and mutation, no changes or masking to reference image are performed. Algorithm learns by comparing difference between generation and reference image.
Results are speed up as details generation can take a while.

![lenaGenPaint](https://user-images.githubusercontent.com/40773550/226498912-39fe0fc7-4c2c-4b32-999b-ecd12b575696.gif)

![brighton](https://user-images.githubusercontent.com/40773550/227058736-05288799-372a-478e-8438-4cf3278cb5fb.jpg)
![brighton jpg](https://user-images.githubusercontent.com/40773550/227059714-6f07d2cd-d3a9-415e-87c9-4f5adc20aed5.gif)

![city](https://user-images.githubusercontent.com/40773550/227060384-e43f812f-5a57-4e06-ae35-2751db01ecf4.jpg)
![city jpg](https://user-images.githubusercontent.com/40773550/227060570-0078867d-45c9-42a3-920d-3fcdc67f8629.gif)

## How to use

```
python3 painter.py Lena.png
```

## Features
- genomes purification - algorithms removes genomes from the pool of all possible combination if they do not produce any improvements, that way algorithm doesn't waste time for trying twice the same solution.