from torch.utils.data import Dataset
import numpy as np
import os
import cv2

class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.IMAGE = os.path.join(self.root, 'IMAGE')
        self.LIGHT = os.path.join(self.root, 'LIGHT')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.ALBEDO = os.path.join(self.root, 'ALBEDO')
        self.TRANSPORT = os.path.join(self.root, 'TRANSPORT')

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.subjects = self.get_subjects()
        self.lights = self.get_lights()

    def get_subjects(self):
        """
        Get the all the training image files' names
        """
        all_subjects = os.listdir(self.IMAGE)

        var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str) # 测试用的数据集
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def get_lights(self):
        return os.listdir(self.LIGHT)

    def __len__(self):
        return len(self.subjects) * len(self.lights)

    def get_item(self, index):
        """
        Traverse all the images of one object in different lights first, then another object.
        """
        light_id = index % len(self.lights)
        subject_id = index // len(self.lights)
        subject = self.subjects[subject_id]

        # Set up file path
        image_path = os.path.join(self.IMAGE, subject, '%03d.jpg' % (light_id))
        mask_path = os.path.join(self.MASK, '%03d.png' % (subject_id)) # for obj in different lights, the mask remains the same
        albedo_path = os.path.join(self.ALBEDO, subject, '%03d.jpg' % (light_id))
        light_path = os.path.join(self.LIGHT, str(self.lights[light_id]))
        transport_path = os.path.join(self.TRANSPORT, subject, '%03d.npy' % (light_id))

        # --------- Read groundtruth file data ------------
        # mask
        # [H, W] bool
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0] != 0

        # image
        # [H, W, 3] 0 ~ 1 float
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        for i in range(image.shape[0]): # mask image
            for j in range(image.shape[1]):
                if not mask[i][j]:
                    image[i][j] = [0, 0, 0]


        # albedo
        # [H, W, 3] 0 ~ 1 float
        albedo = cv2.imread(albedo_path)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB) / 255.0
        for i in range(albedo.shape[0]): # mask albedo
            for j in range(albedo.shape[1]):
                if not mask[i][j]:
                    albedo[i][j] = [0, 0, 0]

        # light
        # [9, 3] SH coefficient
        light = np.load(light_path) # TODO: 进一步cvt

        # transport
        # [H, W, 9]
        transport = np.load(transport_path) # TODO: 进一步cvt
        for i in range(transport.shape[0]): # mask transport
            for j in range(transport.shape[1]):
                if not mask[i][j]:
                    transport[i][j] = [0] * 9

        # flatten
        image = image.reshape((-1, 3))
        mask = mask.reshape((-1))
        albedo = albedo.reshape((-1, 3))
        transport = transport.reshape((-1, 9))

        res = {
            'name': subject,
            'image': image,
            'mask': mask,
            'albedo': albedo,
            'light': light,
            'transport': transport
        }

        return res

    def __getitem__(self, index):
        return self.get_item(index)