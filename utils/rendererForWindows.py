import torch
from torchvision.utils import make_grid
import numpy as np
import cv2
import trimesh


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Adapted to use OpenCV instead of PyOpenGL-based pyrender.
    """
    def __init__(self, img_res=224, faces=None, focal_length=5000):
        self.img_res = img_res
        self.faces = faces
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))
        rend_imgs = []

        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(
                np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2, 0, 1))
            ).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)

        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image):
        # Project 3D vertices onto 2D space using a simple perspective projection
        vertices_2d = self.project_vertices(vertices, camera_translation)

        # Create a blank canvas
        rendered_image = np.zeros((self.img_res, self.img_res, 3), dtype=np.float32)

        # Draw the mesh using OpenCV
        for face in self.faces:
            pts = vertices_2d[face].astype(np.int32)
            cv2.polylines(rendered_image, [pts], isClosed=True, color=(0.8, 0.3, 0.3), thickness=1)

        # Blend the rendered mesh with the input image
        valid_mask = (rendered_image.sum(axis=-1) > 0)[:, :, None].astype(np.float32)
        output_img = (rendered_image * valid_mask + (1 - valid_mask) * image)
        return output_img

    def project_vertices(self, vertices, camera_translation):
        """
        Project 3D vertices to 2D using a perspective projection.
        """
        # Flip the x-axis to match OpenCV's coordinate system
        vertices[:, 0] *= -1

        # Apply camera translation
        vertices = vertices + camera_translation[None, :]

        # Apply perspective projection
        projected = np.zeros((vertices.shape[0], 2), dtype=np.float32)
        projected[:, 0] = self.focal_length * (vertices[:, 0] / vertices[:, 2]) + self.camera_center[0]
        projected[:, 1] = self.focal_length * (vertices[:, 1] / vertices[:, 2]) + self.camera_center[1]
        return projected
