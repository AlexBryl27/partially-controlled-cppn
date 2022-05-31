from PIL import Image
import numpy as np
from sklearn.preprocessing import minmax_scale


class ImageProcessor:

    def __init__(self, reference_path, n_axis=5):
        
        reference = Image.open(reference_path)
        reference = np.array(reference, dtype=np.float32)
        self.nrows, self.ncols = reference.shape[:2]

        if n_axis == 5:
            ones = np.ones((self.nrows, self.ncols, 2), dtype=np.float32)
            reference = np.dstack((reference, ones))
        reference = reference.reshape(-1, n_axis)
        self.reference = minmax_scale(reference, feature_range=(-1, 1))

        self.rowmat = (np.tile(np.linspace(0, self.nrows-1, self.nrows, dtype=np.float32), self.ncols)\
            .reshape(self.ncols, self.nrows).T - self.nrows / 2.0) / (min(self.nrows, self.ncols) / 2.0)
        self.colmat = (np.tile(np.linspace(0, self.ncols-1, self.ncols, dtype=np.float32), self.nrows)\
            .reshape(self.nrows, self.ncols) - self.ncols / 2.0) / (min(self.nrows, self.ncols) / 2.0)

        self.row_strides = [i for i in range(0, self.nrows, self.nrows // 3)][1:]
        self.col_strides = [i for i in range(0, self.ncols, self.ncols // 3)][1:]


    def generate_3d_grid(self):
        
        grid = [
            self.rowmat, 
            self.colmat, 
            np.sqrt(np.power(self.rowmat, 2) + np.power(self.colmat, 2))
            ]
        grid = np.stack(grid).transpose(1, 2, 0).reshape(-1, len(grid))
        grid = minmax_scale(grid, feature_range=(-1, 1))
        
        return grid.astype(np.float32)


    def generate_5d_grid(self):
        
        grid = [
            self.rowmat, 
            self.colmat, 
            np.sqrt(np.power(self.rowmat, 2) + np.power(self.colmat, 2)),
            np.ones(self.rowmat.shape),
            np.ones(self.colmat.shape)
            ]
        grid = np.stack(grid).transpose(1, 2, 0).reshape(-1, len(grid))
        grid = minmax_scale(grid, feature_range=(-1, 1))
        
        return grid.astype(np.float32)

    
    def generate_custom_grid(self, features, cost):
    
        grid = [
            self.rowmat, 
            self.colmat, 
            np.sqrt(np.power(self.rowmat, 2) + np.power(self.colmat, 2))
            ]

        feature_mat = np.ones(self.rowmat.shape)
        
        feature_mat[
            :self.row_strides[0], 
            :self.col_strides[0]
            ] *= features[0]                                # up left
        feature_mat[
            :self.row_strides[0], 
            self.col_strides[0]: self.col_strides[1]
            ] *= features[1]                                # up center 
        feature_mat[
            :self.row_strides[0],
            self.col_strides[1]:
            ] *= features[2]                                # up right
        
        feature_mat[
            self.row_strides[0]: self.row_strides[1], 
            :self.col_strides[0]
            ] *= features[3]                                 # center left
        feature_mat[
            self.row_strides[0]: self.row_strides[1], 
            self.col_strides[1]:
            ] *= features[4]                                 # center right

        feature_mat[
            self.row_strides[1]:, 
            :self.col_strides[0]
            ] *= features[5]                                 # down left
        feature_mat[
            self.row_strides[1]:, 
            self.col_strides[0]: self.col_strides[1]
            ] *= features[6]                                 # down center
        feature_mat[
            self.row_strides[1]:, 
            self.col_strides[1]:
            ] *= features[7]                                 # down right    


        grid.append(feature_mat)
        grid.append(cost * np.ones(self.rowmat.shape))
        grid = np.stack(grid).transpose(1, 2, 0).reshape(-1, len(grid))
        minmax_scale(grid[:, :-2], feature_range=(-1., 1), copy=False)
        return grid.astype(np.float32)


    def get_img_from_arr(self, arr):
        
        # img = minmax_scale(arr)
        img = (1 + arr) / 2
        return (255.0 * img.reshape(self.nrows, self.ncols, img.shape[-1])).astype(np.uint8)
