"""Pyramid "locations": the anchor-free replacement for anchor boxes.

In YOLOv3 every grid cell carries 3 anchor BOXES; in FCOS every grid cell is
just a POINT (a "location") on the input image, and the head regresses the 4
side-distances from that point. This module generates those points.

For a feature map of size (H, W) at stride s, cell (i, j) (row i, column j)
maps back to the input-image pixel

    x = (j + 0.5) * s ,   y = (i + 0.5) * s

i.e. the CENTER of the s x s input patch that cell covers (+0.5 for the half-
cell offset; compare YOLOv3's sigmoid(tx)+cell_index, which learns the offset
instead of fixing it at the center).

CRITICAL ordering invariant: the returned [H*W, 2] tensor is flattened
row-major (y outer, x inner) -- exactly the order produced by flattening a
head output [B, H, W, D] via .reshape(B, H*W, D). The loss and the decoder
both rely on locations[k] matching prediction[:, k, :].
"""

import torch


def make_locations(h: int, w: int, stride: int, device) -> torch.Tensor:
    """Generate the input-image coordinates of every cell on one level.

    Input:
        h, w: feature-map height and width (e.g. 52x52 for stride 8 @ 416).
        stride: this level's stride (8 / 16 / 32).
        device: torch device to create the tensor on.

    Output:
        locations: [H*W, 2] float tensor of (x, y) pixel coordinates at the
            network input scale, row-major order (matches reshape(B, H*W, D)).
    """
    # meshgrid(indexing="ij"): ys[i, j] = i (row index), xs[i, j] = j (column).
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Cell index -> center of its stride x stride input patch.
    xs = (xs.reshape(-1) + 0.5) * stride    # [H*W] input-image x
    ys = (ys.reshape(-1) + 0.5) * stride    # [H*W] input-image y
    return torch.stack([xs, ys], dim=-1)    # [H*W, 2] = (x, y)


# ---- Quick self-test: run this file directly ---------------------------------
# python utils/locations.py
if __name__ == "__main__":
    locs = make_locations(2, 3, 8, device="cpu")
    print("locations for a 2x3 stride-8 map (expected x=4,12,20 / y=4,12):")
    print(locs)
    # Row-major check: first row of cells first (y=4), then the second (y=12).
    expected = torch.tensor([
        [4., 4.], [12., 4.], [20., 4.],
        [4., 12.], [12., 12.], [20., 12.],
    ])
    print("matches expected order:", bool(torch.equal(locs, expected)))
