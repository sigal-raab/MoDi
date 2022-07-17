import unittest
from unittest.mock import MagicMock
from models.stylegan_models import ModulatedConv, EqualConv, Blur
from models.stylegan_models import keep_skeletal_dims
from utils.data import Joint, Edge
from traits import SkeletonAwarePoolTraits, SkeletonAwareConv3DTraits, NonSkeletonAwareTraits
import torch
import random
import numpy as np


device = 'cuda'
blur_kernel = [1, 3, 3, 1]
style_dim = 512
batch = 16
parameter_list = [SkeletonAwarePoolTraits, SkeletonAwareConv3DTraits, NonSkeletonAwareTraits]
entity_list = {traits_class: [Joint] for traits_class in parameter_list}
entity_list[SkeletonAwareConv3DTraits] = [Joint, Edge]
entity_list[SkeletonAwarePoolTraits] = [Joint, Edge]

class MyTestCase(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    #region GENERATE.PY TESTS


    #region GENERATOR TESTS
    def test_generator_keep_dims_shape(self):
        for traits_class in parameter_list:
            for entity_class in entity_list[traits_class]:
                with self.subTest(msg=[traits_class.__name__, entity_class.__name__]):
                    self.generator_keep_dims(traits_class, entity_class)

    def test_generator_upsample_shape(self):
        for traits_class in parameter_list:
            for entity_class in entity_list[traits_class]:
                with self.subTest(msg=[traits_class.__name__, entity_class.__name__]):
                    self.generator_upsample(traits_class, entity_class)

    # def test_generator_keep_dims_values(self):
    #     for traits_class in parameter_list:
    #         for entity_class in entity_list[traits_class]:
    #             with self.subTest(msg=[traits_class.__name__, entity_class.__name__]):
    #                 filename = self.generate_filename('generator_keep_dims', traits_class, entity_class)
    #                 self.generator_keep_dims_values(traits_class, entity_class, filename, create_data=False)

    # def test_generator_upsample_values(self):
    #     for traits_class in parameter_list:
    #         for entity_class in entity_list[traits_class]:
    #             with self.subTest(msg=[traits_class.__name__, entity_class.__name__]):
    #                 filename = self.generate_filename('generator_upsample', traits_class, entity_class)
    #                 self.generator_upsample_values(traits_class, entity_class, filename, create_data=False)

    #endregion

    #region DISCRIMINATOR TESTS
    def test_discriminator_first_and_final_conv_shape(self):
        for traits_class in parameter_list:
            for entity_class in entity_list[traits_class]:
                with self.subTest(msg=[traits_class.__name__, entity_class.__name__]):
                    self.discriminator_first_and_final_conv(traits_class, entity_class)

    def test_discriminator_resBlock_shape(self):
        for traits_class in parameter_list:
            for entity_class in entity_list[traits_class]:
                with self.subTest(msg=[traits_class.__name__, entity_class.__name__]):
                    self.discriminator_resBlock(traits_class, entity_class)

    # def test_discriminator_resBlock_values(self):
    #     for traits_class in parameter_list:
    #         for entity_class in entity_list[traits_class]:
    #             with self.subTest(msg=[traits_class.__name__, entity_class.__name__]):
    #                 filename = self.generate_filename('discriminator_resBlock', traits_class, entity_class)
    #                 weights_mock = self.generate_filename('weights_resblock', traits_class, entity_class)
    #                 self.discriminator_resBlock_values(traits_class, entity_class, filename, create_data=False)

    #endregion

    #region Test methods
    def generator_keep_dims(self, traits_class, entity_class):
        kernel_size = 3
        for i in range(0, len(traits_class.n_channels(entity_class))):
            out_channel = traits_class.n_channels(entity_class)[i]
            skeleton_traits = traits_class(entity_class.parents_list[i], keep_skeletal_dims(traits_class.n_joints(entity_class)[i]))

            conv = ModulatedConv(out_channel, out_channel, kernel_size, style_dim, upsample=False,
                                   blur_kernel=blur_kernel, demodulate=True,
                                   skeleton_traits=skeleton_traits).cuda()

            input = torch.randn(batch, out_channel, skeleton_traits.smaller_n_joints, traits_class.n_frames(entity_class)[i]).to(device)
            style = torch.randn(batch, style_dim).to(device)
            out = conv(input, style)

            self.assertEqual(out.shape, (batch, out_channel, skeleton_traits.smaller_n_joints, traits_class.n_frames(entity_class)[i]))

    def generator_upsample(self, traits_class, entity_class):
        kernel_size = 3
        for i in range(1, traits_class.n_levels(entity_class)):
            in_channel = traits_class.n_channels(entity_class)[i-1]
            out_channel = traits_class.n_channels(entity_class)[i]
            skeleton_traits = traits_class(entity_class.parents_list[i], entity_class.skeletal_pooling_dist_1[i-1])

            conv = ModulatedConv(in_channel, out_channel, kernel_size, style_dim, upsample=True,
                                   blur_kernel=blur_kernel, demodulate=True, skeleton_traits=skeleton_traits).cuda()

            input = torch.randn(batch, in_channel,  skeleton_traits.n_joints(entity_class)[i-1], traits_class.n_frames(entity_class)[i-1] ).to(device)
            style = torch.randn(batch, style_dim).to(device)
            out = conv(input, style)

            self.assertEqual(out.shape, (batch, out_channel, skeleton_traits.n_joints(entity_class)[i], traits_class.n_frames(entity_class)[i]))

    def generator_keep_dims_values(self, traits_class, entity_class, filename, create_data=False):
        self.setUp()
        kernel_size = 3
        i = traits_class.n_levels(entity_class)-2  # one before last
        out_channel = traits_class.n_channels(entity_class)[i]
        skeleton_traits = traits_class(entity_class.parents_list[i], keep_skeletal_dims(traits_class.n_joints(entity_class)[i]))

        conv = ModulatedConv(out_channel, out_channel, kernel_size, style_dim, upsample=False,
                               blur_kernel=blur_kernel, demodulate=True,
                               skeleton_traits=skeleton_traits).cuda()

        if create_data:
            # create test data
            input = torch.randn(batch, out_channel, skeleton_traits.smaller_n_joints, traits_class.n_frames(entity_class)[i]).to(device)
            style = torch.randn(batch, style_dim).to(device)
            save_inputs = {'input': input, 'style': style}
        else:
            # read input from file
            inputs = torch.load(filename)
            input = inputs['input']
            style = inputs['style']

        out = conv(input, style)

        if create_data:
            # create test data
            save_inputs['out'] = out
            torch.save(save_inputs, filename)

        self.assertEqual(out.shape, (batch, out_channel, skeleton_traits.smaller_n_joints, traits_class.n_frames(entity_class)[i]))
        if not create_data:
            self.assertTrue(torch.allclose(out, inputs['out'], atol=1e-04))

    def generator_upsample_values(self, traits_class, entity_class, filename, create_data=False):
        self.setUp()
        kernel_size = 3
        i = traits_class.n_levels(entity_class)-2  # one before last
        in_channel = traits_class.n_channels(entity_class)[i - 1]
        out_channel = traits_class.n_channels(entity_class)[i]
        skeleton_traits = traits_class(entity_class.parents_list[i], entity_class.skeletal_pooling_dist_1[i - 1])

        conv = ModulatedConv(in_channel, out_channel, kernel_size, style_dim, upsample=True,
                               blur_kernel=blur_kernel, demodulate=True, skeleton_traits=skeleton_traits).cuda()
        if create_data:
            # create input data
            input = torch.randn(batch, in_channel,  skeleton_traits.n_joints(entity_class)[i-1], traits_class.n_frames(entity_class)[i-1] ).to(device)
            style = torch.randn(batch, style_dim).to(device)
            save_inputs = {'input': input, 'style': style}
        else:
            # read input from file
            inputs = torch.load(filename)
            input = inputs['input']
            style = inputs['style']

        out = conv(input, style)

        if create_data:
            # create test data
            save_inputs['out'] = out
            torch.save(save_inputs, filename)

        self.assertEqual(out.shape, (batch, out_channel, skeleton_traits.n_joints(entity_class)[i], traits_class.n_frames(entity_class)[i]))
        if not create_data:
            self.assertTrue(torch.allclose(out, inputs['out'], atol=1e-04))

    def discriminator_first_and_final_conv(self, traits_class, entity_class):
        stride = 1
        # First conv
        kernel_size = 1
        skeleton_traits = traits_class(entity_class.parents_list[-1], keep_skeletal_dims(traits_class.n_joints(entity_class)[-1]))
        padding = skeleton_traits.fixed_dim_pad(kernel_size)
        in_channel = entity_class.n_channels
        out_channel = traits_class.n_channels(entity_class)[-1]
        conv = EqualConv(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                           bias=False, skeleton_traits=skeleton_traits).cuda()
        input = torch.randn(batch, in_channel, skeleton_traits.smaller_n_joints, traits_class.n_frames(entity_class)[-1]).to(device)
        out = conv(input)
        self.assertEqual(out.shape, (batch, out_channel, skeleton_traits.smaller_n_joints, traits_class.n_frames(entity_class)[-1]))

        # Final conv
        skeleton_traits = traits_class(entity_class.parents_list[0], keep_skeletal_dims(traits_class.n_joints(entity_class)[0]))
        kernel_size = 3
        padding = skeleton_traits.fixed_dim_pad(kernel_size)
        in_channel = traits_class.n_channels(entity_class)[0]+1
        out_channel = traits_class.n_channels(entity_class)[0]
        conv = EqualConv(in_channel, out_channel, 3, padding=padding, stride=stride,
                           bias=False, skeleton_traits=skeleton_traits).cuda()
        input = torch.randn(batch, traits_class.n_channels(entity_class)[0]+1, skeleton_traits.smaller_n_joints, traits_class.n_frames(entity_class)[-1]).to(device)
        out = conv(input)
        self.assertEqual(out.shape, (batch, traits_class.n_channels(entity_class)[0], skeleton_traits.smaller_n_joints, traits_class.n_frames(entity_class)[-1]))

    def discriminator_resBlock(self, traits_class, entity_class):
        kernel_size = 3
        in_channel = traits_class.n_channels(entity_class)[-1]

        for i in range(traits_class.n_levels(entity_class)-1, 0, -1):
            out_channel = traits_class.n_channels(entity_class)[i-1]

            skeleton_traits_for_kernel_3 = traits_class(entity_class.parents_list[i], entity_class.skeletal_pooling_dist_1[i - 1])
            skeleton_traits_for_kernel_1 = traits_class(entity_class.parents_list[i], entity_class.skeletal_pooling_dist_0[i - 1])
            larger_n_joints = skeleton_traits_for_kernel_3.larger_n_joints

            input = torch.randn(batch, in_channel, skeleton_traits_for_kernel_3.n_joints(entity_class)[i], traits_class.n_frames(entity_class)[i]).to(device)
            input_skip = torch.randn(batch, in_channel, skeleton_traits_for_kernel_3.n_joints(entity_class)[i], traits_class.n_frames(entity_class)[i]).to(device)

            #keep dims
            skeleton_traits_keep_dims = traits_class(skeleton_traits_for_kernel_3.parents,
                                                     keep_skeletal_dims(larger_n_joints))
            stride = 1
            padding = skeleton_traits_keep_dims.fixed_dim_pad(kernel_size)
            conv1 = EqualConv(in_channel, in_channel, kernel_size, padding=padding, stride=stride,
                               bias=False, skeleton_traits=skeleton_traits_keep_dims).cuda()

            out1 = conv1(input)
            self.assertEqual(out1.shape, (batch, in_channel, skeleton_traits_for_kernel_3.n_joints(entity_class)[i], traits_class.n_frames(entity_class)[i]))

            #downsample
            if skeleton_traits_for_kernel_3.need_blur:
                factor = 2
                p = (len(blur_kernel) - factor) + (kernel_size - 1)
                pad0 = (p + 1) // 2
                pad1 = p // 2
                pad = skeleton_traits_for_kernel_3.upfirdn_pad(pad0, pad1)
                blur = Blur(blur_kernel, pad, skeleton_traits=skeleton_traits_for_kernel_3).cuda()
                input = blur(input)

            stride = skeleton_traits_for_kernel_3.updown_stride
            padding = skeleton_traits_for_kernel_3.updown_pad(kernel_size)
            conv2 = EqualConv(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                                bias=False, skeleton_traits=skeleton_traits_for_kernel_3).cuda()

            out2 = conv2(input)
            self.assertEqual(out2.shape, (batch, out_channel, skeleton_traits_for_kernel_3.n_joints(entity_class)[i-1], traits_class.n_frames(entity_class)[i-1]))

            # skip
            stride = skeleton_traits_for_kernel_1.updown_stride
            padding = skeleton_traits_for_kernel_1.updown_pad(1)
            skip = EqualConv(in_channel, out_channel, 1, padding=padding, stride=stride,
                                bias=False, skeleton_traits=skeleton_traits_for_kernel_1).cuda()

            out3 = skip(input_skip)
            self.assertEqual(out3.shape, (batch, out_channel, skeleton_traits_for_kernel_1.n_joints(entity_class)[i-1], traits_class.n_frames(entity_class)[i-1]))

    def discriminator_resBlock_values(self, traits_class, entity_class, filename, create_data=False):
        self.setUp()
        kernel_size = 3

        i = traits_class.n_levels(entity_class)-2  # one before last
        in_channel = traits_class.n_channels(entity_class)[i]
        out_channel = traits_class.n_channels(entity_class)[i-1]

        skeleton_traits_for_kernel_3 = traits_class(entity_class.parents_list[i], entity_class.skeletal_pooling_dist_1[i - 1])
        skeleton_traits_for_kernel_1 = traits_class(entity_class.parents_list[i], entity_class.skeletal_pooling_dist_0[i - 1])
        larger_n_joints = skeleton_traits_for_kernel_3.larger_n_joints

        if create_data:
            # create test data
            input = torch.randn(batch, in_channel, skeleton_traits_for_kernel_3.n_joints(entity_class)[i], traits_class.n_frames(entity_class)[i]).to(device)
            input_skip = torch.randn(batch, in_channel, skeleton_traits_for_kernel_3.n_joints(entity_class)[i], traits_class.n_frames(entity_class)[i]).to(device)
            inputs = {}
            inputs['input']= input
            inputs['input_skip']= input_skip
            self.setUp()

        else:
            #read test data
            inputs = torch.load(filename)
            input = inputs['input']
            input_skip = inputs['input_skip']
            self.setUp()

        #keep dims
        skeleton_traits_keep_dims = traits_class(skeleton_traits_for_kernel_3.parents,
                                                 keep_skeletal_dims(larger_n_joints))
        stride = 1
        padding = skeleton_traits_keep_dims.fixed_dim_pad(kernel_size)

        conv1 = EqualConv(in_channel, in_channel, kernel_size, padding=padding, stride=stride,
                           bias=False, skeleton_traits=skeleton_traits_keep_dims).cuda()

        out1 = conv1(input)

        if create_data:
        # create test data
            inputs['out1'] = out1

        self.assertEqual(out1.shape, (batch, in_channel, skeleton_traits_for_kernel_3.n_joints(entity_class)[i], traits_class.n_frames(entity_class)[i]))
        if not create_data:
            self.assertTrue(torch.allclose(out1, inputs['out1'], atol=1e-04))

        #downsample
        if skeleton_traits_for_kernel_3.need_blur:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            pad = skeleton_traits_for_kernel_3.upfirdn_pad(pad0, pad1)
            blur = Blur(blur_kernel, pad, skeleton_traits=skeleton_traits_for_kernel_3).cuda()
            input = blur(input)

        stride = skeleton_traits_for_kernel_3.updown_stride
        padding = skeleton_traits_for_kernel_3.updown_pad(kernel_size)

        conv2 = EqualConv(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                            bias=False, skeleton_traits=skeleton_traits_for_kernel_3).cuda()

        out2 = conv2(input)

        if create_data:
            # create test data
            inputs['out2'] = out2

        self.assertEqual(out2.shape, (batch, out_channel, skeleton_traits_for_kernel_3.n_joints(entity_class)[i-1], traits_class.n_frames(entity_class)[i-1]))
        if not create_data:
            self.assertTrue(torch.equal(out2, inputs['out2']))

        # skip
        stride = skeleton_traits_for_kernel_1.updown_stride
        padding = skeleton_traits_for_kernel_1.updown_pad(1)

        skip = EqualConv(in_channel, out_channel, 1, padding=padding, stride=stride,
                            bias=False, skeleton_traits=skeleton_traits_for_kernel_1).cuda()

        out3 = skip(input_skip)

        if create_data:
            # create test data
            inputs['out3'] = out3
            torch.save(inputs, filename)

        self.assertEqual(out3.shape, (batch, out_channel, skeleton_traits_for_kernel_1.n_joints(entity_class)[i-1], traits_class.n_frames(entity_class)[i-1]))
        if not create_data:
            self.assertTrue(torch.equal(out3, inputs['out3']))
    #endregion

    #region utilities
    def generate_filename(this, name, traits_class, entity_class):
        filename = f'unittest_data/{name}_{traits_class.__name__}_{entity_class.__name__}.pt'
        return filename
    #endregion


if __name__ == '__main__':
    # unittest.main()
    unittest.debug()
