"""predictor attention knockout — 간선을 끊고 성능 변화를 본다.

  python -m analysis.attention.knockout.run --dataset v11 --model vith \
      --edge mask->ctx --edge ctx->ctx --edge mask->mask --layers each
"""
