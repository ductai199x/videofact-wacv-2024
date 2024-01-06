import torch
from typing import *
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms.functional import resize


class PatchPredictions:
    def __init__(
        self,
        patch_preds: torch.Tensor,
        patch_size: int,
        img_size: Tuple[int, int],
        max_cfs_r=0.55,
        max_nz_r=0.5,
        min_thresh=0.1,
        max_num_regions=2,
        final_thresh=0.27,
    ):
        self.patch_preds = patch_preds
        self.patch_size = patch_size
        self.img_size = img_size
        self.max_cfs_r = max_cfs_r
        self.max_nz_r = max_nz_r
        self.min_thresh = min_thresh
        self.max_num_regions = max_num_regions
        self.final_thresh = final_thresh

        # get the width and height of both the pixel mask and the patch mask
        _, self.pixel_preds_h, self.pixel_preds_w = self.patch_to_pixel(patch_preds)
        self.patch_preds_h, self.patch_preds_w = (
            self.pixel_preds_h // self.patch_size,
            self.pixel_preds_w // self.patch_size,
        )

        self.threshold, self.threshold_cond = self.define_threshold()

    def get_pixel_preds(self):
        # linearly remapping the range to [0, 1] (to normalize + amplify predictions)
        self.patch_preds *= 1 / self.patch_preds.max()

        # set patch predictions < threshold to 0
        self.patch_preds[self.patch_preds < self.threshold] = 0

        # view the predictions as a 2d image
        self.patch_preds = self.patch_preds.view(self.patch_preds_h, self.patch_preds_w)

        # fill in the missing patches as a result of the thresholding step and smooth out the image
        self.fill_in()
        self.patch_preds = torch.Tensor(
            gaussian_filter(self.patch_preds, sigma=0.65, order=0, mode="reflect")
        )
        self.fill_in()

        # eliminate outlying patch predictions
        self.eliminate_outliers()

        # convert from patch predictions to pixel predictions
        # pixel_preds, _, _ = self.patch_to_pixel(self.patch_preds.flatten())
        pixel_preds = resize(self.patch_preds.unsqueeze(0), self.img_size, antialias=False).squeeze(0)

        # threshold this image to get predictions of only 0 and 1
        pixel_preds[pixel_preds > self.final_thresh] = 1.0
        pixel_preds[pixel_preds <= self.final_thresh] = 0.0

        return pixel_preds

    def get_pixel_preds_no_refine(self):
        pixel_preds, _, _ = self.patch_to_pixel(self.patch_preds.flatten())

        return pixel_preds

    def print_parameters(self):
        print(
            f"""
        Patch size: {self.patch_size}
        Max count-from-start ratio: {self.max_cfs_r}
        Max non-zero ratio {self.max_nz_r}
        Minimum threshold {self.min_thresh}
        """
        )

    def pixel_to_patch(self, pixel):
        if len(pixel.shape) == 1:
            pixel.unsqueeze(0)
        batch_size = pixel.shape[0]
        kernel_size, stride = self.patch_size, self.patch_size
        patch = pixel.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        patch = patch.contiguous().view(-1, kernel_size, kernel_size)
        patch = torch.flatten(patch, start_dim=1, end_dim=2)
        patch = torch.sum(patch, dim=1) / (kernel_size * kernel_size)
        patch = patch.view(batch_size, -1)
        return patch

    def patch_to_pixel(self, patch):
        if len(patch.shape) == 1:
            patch.unsqueeze(0)
        batch_size = patch.shape[0]
        kernel_size, stride = self.patch_size, self.patch_size
        # TODO: change this to reflect stride < kernel_size
        pixel_size0, pixel_size1 = (
            kernel_size * (self.img_size[0] // kernel_size),
            kernel_size * (self.img_size[1] // kernel_size),
        )
        pixel = patch.unsqueeze(-1).repeat(1, 1, kernel_size * kernel_size).permute(0, 2, 1)
        pixel = torch.nn.Fold(
            output_size=(pixel_size0, pixel_size1),
            kernel_size=kernel_size,
            stride=stride,
        )(pixel).squeeze()
        return pixel, pixel_size0, pixel_size1

    def define_threshold(self):
        patch_prob = self.patch_preds[self.patch_preds > self.min_thresh].flatten()
        bins = torch.arange(0.1, 1.0, 0.05)
        histogram = patch_prob.histogram(bins=bins, density=False)
        dist = histogram.hist / histogram.hist.sum()
        dist = [0.0] + list(dist)

        start = -1
        end = -1
        count_from_start = 0
        num_non_zero = 0
        max_cfs = len(bins) * self.max_cfs_r
        max_nz = len(bins) * self.max_nz_r
        cond = "end of loop"
        for b, v in zip(list(bins)[::-1], dist[::-1]):
            end = b
            if start == -1 and v > 1e-10:
                start = b
            if start != -1:
                count_from_start += 1
                if v > 0:
                    num_non_zero += 1
                if count_from_start >= max_cfs:
                    return end, "count_from_start"
                if num_non_zero >= max_nz:
                    return end, "num_non_zero"
        return end, cond

    def eliminate_outliers(self):
        # setup temporary buffers (we don't want to modify the original)
        h, w = self.patch_preds_h, self.patch_preds_w
        patch_temp = torch.clone(self.patch_preds)
        region_vs_connectivity = []

        region = []
        max_connected = -1

        # recursive floodfill algorithm
        def floodfill(i, j):
            if i >= h or j >= w or patch_temp[i][j] == -1 or patch_temp[i][j] == 0 or i < 0 or j < 0:
                return 0
            else:
                region.append((i, j))
                patch_temp[i][j] = -1
                return 1 + (
                    floodfill(i + 1, j)
                    + floodfill(i - 1, j)
                    + floodfill(i, j + 1)
                    + floodfill(i, j - 1)
                    + floodfill(i + 1, j - 1)
                    + floodfill(i + 1, j + 1)
                    + floodfill(i - 1, j - 1)
                    + floodfill(i - 1, j + 1)
                )

        for i in range(h):
            for j in range(w):
                ff = floodfill(i, j)
                region_vs_connectivity.append((ff, region))
                region = []  # reset for next iteration

        # if no regions are detected -> return. else, set everything patch(i,j) not in max_region to 0
        if len(region_vs_connectivity) == 0:
            print("WARNING: No regions detected.")
            return
        else:
            region_vs_connectivity = sorted(region_vs_connectivity, key=lambda x: x[0], reverse=True)
            nodes_to_keep = []
            for _, r in region_vs_connectivity[: self.max_num_regions]:
                nodes_to_keep += r

            # remove all nodes not in nodes_to_keep
            for i in range(h):
                for j in range(w):
                    if (i, j) not in nodes_to_keep:
                        self.patch_preds[i, j] = 0

    def fill_in(self):
        h, w = self.patch_preds_h, self.patch_preds_w
        fill_idx = []
        for i in range(h):
            for j in range(w):
                left = 0 if j - 1 < 0 else j - 1
                top = 0 if i - 1 < 0 else i - 1
                right = w if j + 1 > w else j + 1
                bottom = h if i + 1 > h else i + 1
                num_connected = self.patch_preds[top : bottom + 1, left : right + 1].sum()
                surround_area = (right - left + 1) * (bottom - top + 1) - 1  # -1 for patch(i,j) itself
                if num_connected > surround_area * 0.6:
                    if self.patch_preds[i, j] <= surround_area * 0.6:
                        fill_idx.append((i, j, num_connected / surround_area))

        for i, j, a in fill_idx:
            self.patch_preds[i, j] = a
